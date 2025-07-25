//! Async executors.
//!
//! 异步执行器。
//!
//! This crate provides two reference executors that trade performance for
//! functionality. They should be considered reference executors that are "good
//! enough" for most use cases. For more specialized use cases, consider writing
//! your own executor on top of [`async-task`].
//!
//! 此包提供了2个参考的执行器，在性能和功能上做了权衡。
//! - Executor
//! - LocalExecutor
//!
//! 这两个执行器在多数情况下已经足够好。
//! 对于一些额数的场景，考虑自己在[`async-task`]之上编写执行器。
//!
//! [`async-task`]: https://crates.io/crates/async-task
//!
//! # Examples
//!
//! ```
//! use async_executor::Executor;
//! use futures_lite::future;
//!
//! // Create a new executor.
//! let ex = Executor::new();
//!
//! // Spawn a task.
//! let task = ex.spawn(async {
//!     println!("Hello world");
//! });
//!
//! // Run the executor until the task completes.
//! future::block_on(ex.run(task));
//! ```

#![warn(
    missing_docs,
    missing_debug_implementations,
    rust_2018_idioms,
    clippy::undocumented_unsafe_blocks
)]
#![doc(
    html_favicon_url = "https://raw.githubusercontent.com/smol-rs/smol/master/assets/images/logo_fullsize_transparent.png"
)]
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/smol-rs/smol/master/assets/images/logo_fullsize_transparent.png"
)]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]

use std::fmt;
use std::marker::PhantomData;
use std::panic::{RefUnwindSafe, UnwindSafe};
use std::pin::Pin;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, RwLock, TryLockError};
use std::task::{Context, Poll, Waker};

use async_task::{Builder, Runnable};
use concurrent_queue::ConcurrentQueue;
use futures_lite::{future, prelude::*};
use pin_project_lite::pin_project;
use slab::Slab;

#[cfg(feature = "static")]
mod static_executors;

#[doc(no_inline)]
pub use async_task::{FallibleTask, Task};
#[cfg(feature = "static")]
#[cfg_attr(docsrs, doc(cfg(any(feature = "static"))))]
pub use static_executors::*;

/// An async executor.
/// 异步执行器。
///
/// # Examples
///
/// A multi-threaded executor:
///
/// ```
/// use async_channel::unbounded;
/// use async_executor::Executor;
/// use easy_parallel::Parallel;
/// use futures_lite::future;
///
/// let ex = Executor::new();
/// let (signal, shutdown) = unbounded::<()>();
///
/// Parallel::new()
///     // Run four executor threads.
///     .each(0..4, |_| future::block_on(ex.run(shutdown.recv())))
///     // Run the main future on the current thread.
///     .finish(|| future::block_on(async {
///         println!("Hello world!");
///         drop(signal);
///     }));
/// ```
pub struct Executor<'a> {
    /// The executor state.
    /// 执行器状态：
    /// - 维护了裸指针`*mut State`指向状态内存块，可以按需还原出内容，同时不受所有权限制。
    /// - 为什么用原子操作：因为需要同时在多个线程访问。
    state: AtomicPtr<State>,

    /// Makes the `'a` lifetime invariant.
    /// 隐含的引用了具有内部可变性的State对象。
    _marker: PhantomData<std::cell::UnsafeCell<&'a ()>>,
}

// SAFETY: Executor stores no thread local state that can be accessed via other thread.
// 执行器可以在线程间移动，没有与线程绑定的信息。
unsafe impl Send for Executor<'_> {}
// SAFETY: Executor internally synchronizes all of it's operations internally.
// 安全性：执行器自身负责处理所有同步。
unsafe impl Sync for Executor<'_> {}

impl UnwindSafe for Executor<'_> {}
impl RefUnwindSafe for Executor<'_> {}

impl fmt::Debug for Executor<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        debug_executor(self, "Executor", f)
    }
}

impl<'a> Executor<'a> {
    /// Creates a new executor.
    ///
    /// 创建一个执行器。此时执行器状态还未初始化。
    ///
    /// # Examples
    ///
    /// ```
    /// use async_executor::Executor;
    ///
    /// let ex = Executor::new();
    /// ```
    pub const fn new() -> Executor<'a> {
        Executor {
            state: AtomicPtr::new(std::ptr::null_mut()),
            _marker: PhantomData,
        }
    }

    /// Returns `true` if there are no unfinished tasks.
    /// 查看是否还有未完成的任务。(即活跃任务表中任务数为0)。
    ///
    /// # Examples
    ///
    /// ```
    /// use async_executor::Executor;
    ///
    /// let ex = Executor::new();
    /// assert!(ex.is_empty());
    ///
    /// let task = ex.spawn(async {
    ///     println!("Hello world");
    /// });
    /// assert!(!ex.is_empty());
    ///
    /// assert!(ex.try_tick());
    /// assert!(ex.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.state().active().is_empty()
    }

    /// Spawns a task onto the executor.
    /// 在执行器上孵化一个任务。需要获取内部内部状态中的活跃任务锁，以登记活跃任务。
    ///
    /// # Examples
    ///
    /// ```
    /// use async_executor::Executor;
    ///
    /// let ex = Executor::new();
    ///
    /// let task = ex.spawn(async {
    ///     println!("Hello world");
    /// });
    /// ```
    pub fn spawn<T: Send + 'a>(&self, future: impl Future<Output = T> + Send + 'a) -> Task<T> {
        let mut active = self.state().active();

        // SAFETY: `T` and the future are `Send`.
        unsafe { self.spawn_inner(future, &mut active) }
    }

    /// Spawns many tasks onto the executor.
    ///
    /// 在执行器上孵化一批任务。只需要获取一次活跃任务锁，任务量大时可以提升性能。
    ///
    /// As opposed to the [`spawn`] method, this locks the executor's inner task lock once and
    /// spawns all of the tasks in one go. With large amounts of tasks this can improve
    /// contention.
    ///
    /// For very large numbers of tasks the lock is occasionally dropped and re-acquired to
    /// prevent runner thread starvation. It is assumed that the iterator provided does not
    /// block; blocking iterators can lock up the internal mutex and therefore the entire
    /// executor.
    ///
    /// ## Example
    ///
    /// ```
    /// use async_executor::Executor;
    /// use futures_lite::{stream, prelude::*};
    /// use std::future::ready;
    ///
    /// # futures_lite::future::block_on(async {
    /// let mut ex = Executor::new();
    ///
    /// let futures = [
    ///     ready(1),
    ///     ready(2),
    ///     ready(3)
    /// ];
    ///
    /// // Spawn all of the futures onto the executor at once.
    /// let mut tasks = vec![];
    /// ex.spawn_many(futures, &mut tasks);
    ///
    /// // Await all of them.
    /// let results = ex.run(async move {
    ///     stream::iter(tasks).then(|x| x).collect::<Vec<_>>().await
    /// }).await;
    /// assert_eq!(results, [1, 2, 3]);
    /// # });
    /// ```
    ///
    /// [`spawn`]: Executor::spawn
    pub fn spawn_many<T: Send + 'a, F: Future<Output = T> + Send + 'a>(
        &self,
        futures: impl IntoIterator<Item = F>,
        handles: &mut impl Extend<Task<F::Output>>,
    ) {
        // 获取活跃任务锁。
        let mut active = Some(self.state().active());

        // Convert the futures into tasks.
        // 使用迭代的方式传入一组furues，传出一组孵化任务的句柄。
        let tasks = futures.into_iter().enumerate().map(move |(i, future)| {
            // SAFETY: `T` and the future are `Send`.
            let task = unsafe { self.spawn_inner(future, active.as_mut().unwrap()) };

            // Yield the lock every once in a while to ease contention.
            // 每500个任务释放一次锁给别人，然后再获取。
            if i.wrapping_sub(1) % 500 == 0 {
                drop(active.take());
                active = Some(self.state().active());
            }

            task
        });

        // Push the tasks to the user's collection.
        // 任务句柄追加到句柄列表。
        handles.extend(tasks);
    }

    /// Spawn a future while holding the inner lock.
    ///
    /// 在已经持有活跃任务锁的情况下，孵化一个任务。
    ///
    /// # Safety
    ///
    /// If this is an `Executor`, `F` and `T` must be `Send`.
    unsafe fn spawn_inner<T: 'a>(
        &self,
        future: impl Future<Output = T> + 'a,
        active: &mut Slab<Waker>,
    ) -> Task<T> {
        // Remove the task from the set of active tasks when the future finishes.
        // 在活跃任务表中获取一个空闲槽位。
        let entry = active.vacant_entry();
        let index = entry.key();
        let state = self.state_as_arc();
        // 在future被遗弃时，自动将任务从活跃任务表中清除。
        let future = AsyncCallOnDrop::new(future, move || drop(state.active().try_remove(index)));

        // Create the task and register it in the set of active tasks.
        // 创建任务并将任务的唤醒器注册到活跃任务表。
        //
        // SAFETY:
        //
        // If `future` is not `Send`, this must be a `LocalExecutor` as per this
        // function's unsafe precondition. Since `LocalExecutor` is `!Sync`,
        // `try_tick`, `tick` and `run` can only be called from the origin
        // thread of the `LocalExecutor`. Similarly, `spawn` can only  be called
        // from the origin thread, ensuring that `future` and the executor share
        // the same origin thread. The `Runnable` can be scheduled from other
        // threads, but because of the above `Runnable` can only be called or
        // dropped on the origin thread.
        //
        // `future` is not `'static`, but we make sure that the `Runnable` does
        // not outlive `'a`. When the executor is dropped, the `active` field is
        // drained and all of the `Waker`s are woken. Then, the queue inside of
        // the `Executor` is drained of all of its runnables. This ensures that
        // runnables are dropped and this precondition is satisfied.
        //
        // `self.schedule()` is `Send`, `Sync` and `'static`, as checked below.
        // Therefore we do not need to worry about what is done with the
        // `Waker`.
        let (runnable, task) = Builder::new()
            .propagate_panic(true)
            .spawn_unchecked(|()| future, self.schedule());
        entry.insert(runnable.waker());

        // 任务创建好后，立即调度一次。
        runnable.schedule();
        task
    }

    /// Attempts to run a task if at least one is scheduled.
    ///
    /// 尝试获取并轮询一个已调度任务。（一般POLL一次称为一个异步时钟)
    /// - 如果没有找到任务返回fale。
    /// - 找到任务返回true。
    ///
    /// Running a scheduled task means simply polling its future once.
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// use async_executor::Executor;
    ///
    /// let ex = Executor::new();
    /// assert!(!ex.try_tick()); // no tasks to run
    ///
    /// let task = ex.spawn(async {
    ///     println!("Hello world");
    /// });
    /// assert!(ex.try_tick()); // a task was found
    /// ```
    pub fn try_tick(&self) -> bool {
        self.state().try_tick()
    }

    /// Runs a single task.<br>
    /// 异步获取并轮询一个已调度任务。
    /// - 如果没找到任务，就异步等待直到有一个。
    /// - 如果找到了就轮询它一次。
    ///
    /// Running a task means simply polling its future once.
    ///
    /// If no tasks are scheduled when this method is called, it will wait until one is scheduled.
    ///
    /// # Examples
    ///
    /// ```
    /// use async_executor::Executor;
    /// use futures_lite::future;
    ///
    /// let ex = Executor::new();
    ///
    /// let task = ex.spawn(async {
    ///     println!("Hello world");
    /// });
    /// future::block_on(ex.tick()); // runs the task
    /// ```
    pub async fn tick(&self) {
        self.state().tick().await;
    }

    /// Runs the executor until the given future completes.
    ///
    /// 启动一个工人。传入的异步操作会与工人守护任务串联。
    ///
    /// # Examples
    ///
    /// ```
    /// use async_executor::Executor;
    /// use futures_lite::future;
    ///
    /// let ex = Executor::new();
    ///
    /// let task = ex.spawn(async { 1 + 2 });
    /// let res = future::block_on(ex.run(async { task.await * 2 }));
    ///
    /// assert_eq!(res, 6);
    /// ```
    pub async fn run<T>(&self, future: impl Future<Output = T>) -> T {
        self.state().run(future).await
    }

    /// Returns a function that schedules a runnable task when it gets woken up.
    ///
    /// 构造一个默认的任务调度器(一个闭包)
    /// - 捕获了执行器状态state的引用。(所谓调度就是将任务插入调度队列)
    /// - 调度流程：将任务插入全局队列，并通知执行器。
    ///
    /// 后续改进：
    /// - 可以尝试如何将其直接写进来源工人的本地队列。
    fn schedule(&self) -> impl Fn(Runnable) + Send + Sync + 'static {
        let state = self.state_as_arc();

        // TODO: If possible, push into the current local queue and notify the ticker.
        move |runnable| {
            state.queue.push(runnable).unwrap();
            state.notify();
        }
    }

    /// Returns a pointer to the inner state.<br>
    /// 执行器状态指针。(首次调用时初始化)
    #[inline]
    fn state_ptr(&self) -> *const State {
        #[cold]
        fn alloc_state(atomic_ptr: &AtomicPtr<State>) -> *mut State {
            let state = Arc::new(State::new());
            // TODO: Switch this to use cast_mut once the MSRV can be bumped past 1.65
            let ptr = Arc::into_raw(state) as *mut State;
            if let Err(actual) = atomic_ptr.compare_exchange(
                std::ptr::null_mut(),
                ptr,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                // SAFETY: This was just created from Arc::into_raw.
                // state = Arc<State> 在上面被移走了，必须手动回收一个计数。
                drop(unsafe { Arc::from_raw(ptr) });
                actual
            } else {
                ptr
            }
        }

        let mut ptr = self.state.load(Ordering::Acquire);
        if ptr.is_null() {
            ptr = alloc_state(&self.state);
        }
        ptr
    }

    /// Returns a reference to the inner state.<br>
    /// 执行器状态的引用。(通过重引用获取)
    #[inline]
    fn state(&self) -> &State {
        // SAFETY: So long as an Executor lives, it's state pointer will always be valid
        // when accessed through state_ptr.
        unsafe { &*self.state_ptr() }
    }

    /// Clones the inner state Arc.<br>
    /// 获取执行器状态的Arc智能指针。会创建一个原始的Arc并将其泄露。
    #[inline]
    fn state_as_arc(&self) -> Arc<State> {
        // SAFETY: So long as an Executor lives, it's state pointer will always be a valid
        // Arc when accessed through state_ptr.
        // 安全约定：
        // 被Arc计数归零后，会把Arc的内容遗弃，但是只要Executor存活就必须保证state指针有效性。
        // 因此，为了防止创建出的Arc计数归零后被内容遗弃，应该将其计数人工加一，永远不会归零。
        let arc = unsafe { Arc::from_raw(self.state_ptr()) };
        let clone = arc.clone();
        std::mem::forget(arc);
        clone
    }
}

/// 遗弃执行器。实际是遗弃内部状态。
///
/// 步骤：
/// - 如果内部状态为空指针，直接返回
/// - 否则：
///   - 将所有任务都最后唤醒一次
///   - 清空全局队列
impl Drop for Executor<'_> {
    fn drop(&mut self) {
        let ptr = *self.state.get_mut();
        if ptr.is_null() {
            return;
        }

        // SAFETY: As ptr is not null, it was allocated via Arc::new and converted
        // via Arc::into_raw in state_ptr.
        let state = unsafe { Arc::from_raw(ptr) };

        let mut active = state.active();
        for w in active.drain() {
            w.wake();
        }
        drop(active);

        while state.queue.pop().is_ok() {}
    }
}

impl<'a> Default for Executor<'a> {
    fn default() -> Executor<'a> {
        Executor::new()
    }
}

/// A thread-local executor.<br>
/// 一个线程本地执行器。
///
/// The executor can only be run on the thread that created it.
/// 执行器只能运行于创建它的线程。(非Send，非Sync)。
///
/// 实现：
/// - 包装了一个普通的执行器，包装抹去其Send和Sync实现，将其局限于当前线程。
/// - 与Executor区别：孵化的任务Future不需要Send。执行器被限于当前线程，所执行的任务也无需跨线程。
///
/// 使用：
/// - 可以启动多个线程，每个线程启动自己的LocalExecutor实例。实现类似Thread-Per-Core的效果。
///
/// 优势：
/// - 启动一个工人，没有等待状态锁、全局队列锁的开销。
/// - 任务只在当前线程调度，减小了cpu上下文切换的开销。
///
/// # Examples
///
/// ```
/// use async_executor::LocalExecutor;
/// use futures_lite::future;
///
/// let local_ex = LocalExecutor::new();
///
/// future::block_on(local_ex.run(async {
///     println!("Hello world!");
/// }));
/// ```
pub struct LocalExecutor<'a> {
    /// The inner executor.
    inner: Executor<'a>,

    /// Makes the type `!Send` and `!Sync`.
    _marker: PhantomData<Rc<()>>,
}

impl UnwindSafe for LocalExecutor<'_> {}
impl RefUnwindSafe for LocalExecutor<'_> {}

impl fmt::Debug for LocalExecutor<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        debug_executor(&self.inner, "LocalExecutor", f)
    }
}

impl<'a> LocalExecutor<'a> {
    /// Creates a single-threaded executor.
    ///
    /// # Examples
    ///
    /// ```
    /// use async_executor::LocalExecutor;
    ///
    /// let local_ex = LocalExecutor::new();
    /// ```
    pub const fn new() -> LocalExecutor<'a> {
        LocalExecutor {
            inner: Executor::new(),
            _marker: PhantomData,
        }
    }

    /// Returns `true` if there are no unfinished tasks.
    ///
    /// # Examples
    ///
    /// ```
    /// use async_executor::LocalExecutor;
    ///
    /// let local_ex = LocalExecutor::new();
    /// assert!(local_ex.is_empty());
    ///
    /// let task = local_ex.spawn(async {
    ///     println!("Hello world");
    /// });
    /// assert!(!local_ex.is_empty());
    ///
    /// assert!(local_ex.try_tick());
    /// assert!(local_ex.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.inner().is_empty()
    }

    /// Spawns a task onto the executor.
    ///
    /// # Examples
    ///
    /// ```
    /// use async_executor::LocalExecutor;
    ///
    /// let local_ex = LocalExecutor::new();
    ///
    /// let task = local_ex.spawn(async {
    ///     println!("Hello world");
    /// });
    /// ```
    pub fn spawn<T: 'a>(&self, future: impl Future<Output = T> + 'a) -> Task<T> {
        let mut active = self.inner().state().active();

        // SAFETY: This executor is not thread safe, so the future and its result
        //         cannot be sent to another thread.
        unsafe { self.inner().spawn_inner(future, &mut active) }
    }

    /// Spawns many tasks onto the executor.
    ///
    /// As opposed to the [`spawn`] method, this locks the executor's inner task lock once and
    /// spawns all of the tasks in one go. With large amounts of tasks this can improve
    /// contention.
    ///
    /// It is assumed that the iterator provided does not block; blocking iterators can lock up
    /// the internal mutex and therefore the entire executor. Unlike [`Executor::spawn`], the
    /// mutex is not released, as there are no other threads that can poll this executor.
    ///
    /// ## Example
    ///
    /// ```
    /// use async_executor::LocalExecutor;
    /// use futures_lite::{stream, prelude::*};
    /// use std::future::ready;
    ///
    /// # futures_lite::future::block_on(async {
    /// let mut ex = LocalExecutor::new();
    ///
    /// let futures = [
    ///     ready(1),
    ///     ready(2),
    ///     ready(3)
    /// ];
    ///
    /// // Spawn all of the futures onto the executor at once.
    /// let mut tasks = vec![];
    /// ex.spawn_many(futures, &mut tasks);
    ///
    /// // Await all of them.
    /// let results = ex.run(async move {
    ///     stream::iter(tasks).then(|x| x).collect::<Vec<_>>().await
    /// }).await;
    /// assert_eq!(results, [1, 2, 3]);
    /// # });
    /// ```
    ///
    /// [`spawn`]: LocalExecutor::spawn
    /// [`Executor::spawn_many`]: Executor::spawn_many
    pub fn spawn_many<T: 'a, F: Future<Output = T> + 'a>(
        &self,
        futures: impl IntoIterator<Item = F>,
        handles: &mut impl Extend<Task<F::Output>>,
    ) {
        let mut active = self.inner().state().active();

        // Convert all of the futures to tasks.
        let tasks = futures.into_iter().map(|future| {
            // SAFETY: This executor is not thread safe, so the future and its result
            //         cannot be sent to another thread.
            unsafe { self.inner().spawn_inner(future, &mut active) }

            // As only one thread can spawn or poll tasks at a time, there is no need
            // to release lock contention here.
        });

        // Push them to the user's collection.
        handles.extend(tasks);
    }

    /// Attempts to run a task if at least one is scheduled.
    ///
    /// Running a scheduled task means simply polling its future once.
    ///
    /// # Examples
    ///
    /// ```
    /// use async_executor::LocalExecutor;
    ///
    /// let ex = LocalExecutor::new();
    /// assert!(!ex.try_tick()); // no tasks to run
    ///
    /// let task = ex.spawn(async {
    ///     println!("Hello world");
    /// });
    /// assert!(ex.try_tick()); // a task was found
    /// ```
    pub fn try_tick(&self) -> bool {
        self.inner().try_tick()
    }

    /// Runs a single task.
    ///
    /// Running a task means simply polling its future once.
    ///
    /// If no tasks are scheduled when this method is called, it will wait until one is scheduled.
    ///
    /// # Examples
    ///
    /// ```
    /// use async_executor::LocalExecutor;
    /// use futures_lite::future;
    ///
    /// let ex = LocalExecutor::new();
    ///
    /// let task = ex.spawn(async {
    ///     println!("Hello world");
    /// });
    /// future::block_on(ex.tick()); // runs the task
    /// ```
    pub async fn tick(&self) {
        self.inner().tick().await
    }

    /// Runs the executor until the given future completes.
    ///
    /// # Examples
    ///
    /// ```
    /// use async_executor::LocalExecutor;
    /// use futures_lite::future;
    ///
    /// let local_ex = LocalExecutor::new();
    ///
    /// let task = local_ex.spawn(async { 1 + 2 });
    /// let res = future::block_on(local_ex.run(async { task.await * 2 }));
    ///
    /// assert_eq!(res, 6);
    /// ```
    pub async fn run<T>(&self, future: impl Future<Output = T>) -> T {
        self.inner().run(future).await
    }

    /// Returns a reference to the inner executor.
    fn inner(&self) -> &Executor<'a> {
        &self.inner
    }
}

impl<'a> Default for LocalExecutor<'a> {
    fn default() -> LocalExecutor<'a> {
        LocalExecutor::new()
    }
}

/// The state of a executor.
/// 执行器状态数据。
struct State {
    /// The global queue.
    /// 全局任务队列。
    queue: ConcurrentQueue<Runnable>,

    /// Local queues created by runners.
    /// 所有工人的本地队列汇总。
    local_queues: RwLock<Vec<Arc<ConcurrentQueue<Runnable>>>>,

    /// Set to `true` when a sleeping ticker is notified or no tickers are sleeping.
    /// 是一个互斥标记，表明当前是否有工人正在执行自我清醒逻辑。
    /// - 刷新逻辑：Sleepers::count != Sleepers::wakers::len()
    /// - 刷新时机：工人进入睡眠、进入清醒、遗弃之后，或Executor被通知前。
    notified: AtomicBool,

    /// A list of sleeping tickers.
    /// 工人休眠管理器。记录了所有工人所依附的线程的恢复器。
    sleepers: Mutex<Sleepers>,

    /// Currently active tasks.
    /// 活动任务表。记录了所有任务的唤醒器。
    active: Mutex<Slab<Waker>>,
}

impl State {
    /// Creates state for a new executor.
    /// 创建执行器状态数据。
    const fn new() -> State {
        State {
            queue: ConcurrentQueue::unbounded(),
            local_queues: RwLock::new(Vec::new()),
            notified: AtomicBool::new(true),
            sleepers: Mutex::new(Sleepers {
                count: 0,
                wakers: Vec::new(),
                free_ids: Vec::new(),
            }),
            active: Mutex::new(Slab::new()),
        }
    }

    /// Returns a reference to currently active tasks.
    /// 锁定活动任务表。锁中毒的情况下强行锁定。
    fn active(&self) -> MutexGuard<'_, Slab<Waker>> {
        self.active.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Notifies a sleeping ticker.
    ///
    /// 有新任务就绪了，通知执行器取出并执行之。
    /// - 1.前提：notified=false。
    /// - 2.动作：从工人睡眠管理器中获取一个线程恢复器，执行工人自我清醒流程。
    #[inline]
    fn notify(&self) {
        if self
            .notified
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let waker = self.sleepers.lock().unwrap().notify();
            if let Some(w) = waker {
                w.wake();
            }
        }
    }

    /// 尝试从全局队列取一个任务并轮询一次。
    ///
    /// 如果成功取到一个任务，则：
    /// 1.传播其它工人唤醒。
    /// 2.轮询一次取出的任务，并返回true。
    ///
    /// 否则，返回false。
    pub(crate) fn try_tick(&self) -> bool {
        match self.queue.pop() {
            Err(_) => false,
            Ok(runnable) => {
                // Notify another ticker now to pick up where this ticker left off, just in case
                // running the task takes a long time.
                self.notify();

                // Run the task.
                runnable.run();
                true
            }
        }
    }

    /// 异步的轮询全局队列中的一个任务。
    ///
    /// - 新建一个任务搜索器
    /// - 异步的从全局队列获取一个已调度任务，获取不到则等待。
    /// - 获取到任务后，轮询一次任务。
    pub(crate) async fn tick(&self) {
        let runnable = Ticker::new(self).runnable().await;
        runnable.run();
    }

    /// 启动带任务窃取功能的工人。
    ///
    /// 主线逻辑：
    /// - 启动一个常驻执行任务的协程。
    /// - 如果传入的future完成了则直接返回。
    ///
    /// 暂时让出：
    /// - 每执行100次任务，让出线程给同线程的其它协程，这里是future参数。
    /// - 如果loop中一直有任务需要执行，则future没有执行机会。
    ///
    /// 随机数生成器实例rng：
    /// - 窃取任务时，被窃取的任务序列依据生成的随机数进行翻转，改变执行次序。
    ///
    /// 工人外壳函数：
    /// - 外壳函数是一个常驻循环的、永远不会终结的异步任务。
    /// - 外壳函数的职责是，不断的、异步的让工人异步获取已就绪任务，然后执行任务。
    /// - 当获取不到已就绪任务时，则异步等待，同时工人进入睡眠状态，工人线程暂停。
    /// - 异步唤醒：当有任务就绪时，执行器发送通知，唤醒工人线程和工人继续异步干活。
    pub async fn run<T>(&self, future: impl Future<Output = T>) -> T {
        let mut runner = Runner::new(self);
        let mut rng = fastrand::Rng::new();

        // A future that runs tasks forever.
        let run_forever = async {
            loop {
                for _ in 0..200 {
                    let runnable = runner.runnable(&mut rng).await;
                    runnable.run();
                }
                future::yield_now().await;
            }
        };

        // Run `future` and `run_forever` concurrently until `future` completes.
        // 同时运行future和run_forever操作，直到有一个完成则返回。
        future.or(run_forever).await
    }
}

/// A list of sleeping tickers.
/// 工人睡眠管理。
///
/// 概念：
/// - 工人：获取已就绪任务并执行之的一段常驻程序。
/// - 工人线程：工人所占据的线程。
/// - 工人时钟：Ticker，可以看做工人的一部分，负责代理工人的睡眠和找任务等工作。
/// - 工人状态：
///   * 睡眠：Ticker::sleeping == 0。
///   * 清醒：Ticker::sleeping > 0，Waker in Sleepers::sleppers，线程暂停。
/// - 工人线程：
///   * 线程暂停
///   * 线程恢复器：用于恢复工人线程，进而恢复工人的新一轮执行
/// - 工人被通知：
///   * 当有已就绪任务时，需要工人来执行，因此向一个睡眠中的工人发送通知，工人进入自我清醒流程。
///
/// 工人的自我清醒流程：
/// - 通知工人：执行器向休眠管理器发起通知。此时工人进入被通知状态。
/// - 恢复线程：休眠管理器取出休眠工人对应的线程恢复器，恢复工人所在线程。
/// - 查找任务：工人尝试获取任务。
/// - 工人清醒：如果获取成功了，则工人标记为清醒。(继续传播通知以唤醒其它工人)
/// - 保持休眠：如果没有获取到任务，则工人继续保持睡眠状态，线程也同时进入暂停状态。
///
/// 注意：
/// - 工人的睡眠与线程的暂停并非完全重合的。
/// - 工人被通知后线程先恢复执行，尝试获取任务并自我清醒，(此过程中工人依然是休眠状态)。
/// - 工人如果成功获取到了任务，此时才会标记为清醒状态(sleeping=0)，否则工人继续标记为睡眠中。
///
/// 关于count值与wakers.len()：
/// - 被通知后，count值并不会立即变化，只有后续工人获取到任务成功清醒后，才会减小count值。
/// - 被通知后，wakers会弹出一个，因此wakers.len()会减一。
/// - 工人开始睡眠时，会增加count值，也会插入到wakers，因此两者同时增加。
/// - 结论：count值始终与sleepint=0的Ticker数量相等，即始终等于睡眠中的工人数量。(包括已被通知还未唤醒成功的)
/// - 结论：当count != wakers.len() 时，说明此时正有被通知的工人(线程恢复了)尝试获取任务让自己变清醒。
struct Sleepers {
    /// Number of sleeping tickers (both notified and unnotified).
    /// 睡眠中的工人数量。即sleeping=0的工人数量。
    count: usize,

    /// IDs and wakers of sleeping unnotified tickers.
    ///
    /// 睡眠中的工人对应的线程恢复器。
    ///
    /// A sleeping ticker is notified when its waker is missing from this list.<br>
    wakers: Vec<(usize, Waker)>,

    /// Reclaimed IDs.
    ///
    /// 可被复用的睡眠ID列表。
    free_ids: Vec<usize>,
}

impl Sleepers {
    /// Inserts a new sleeping ticker.
    /// 插入一个睡眠工人的线程恢复器，休眠计数加1。
    ///
    /// 调用时机：
    /// - 当对清醒状态的工人执行Ticker::sleep。
    ///
    /// 确认线程恢复器的ID:
    /// - 如果有空闲ID，则复用空闲ID;(唤醒器移除后ID会放入空闲列表)
    /// - 如果没有空闲ID，则ID=count+1;
    fn insert(&mut self, waker: &Waker) -> usize {
        let id = match self.free_ids.pop() {
            Some(id) => id,
            None => self.count + 1,
        };
        self.count += 1;
        self.wakers.push((id, waker.clone()));
        id
    }

    /// Re-inserts a sleeping ticker's waker if it was notified.<br>
    /// 更新休眠工人的线程恢复器。
    ///
    /// 调用时机：
    /// - 当对睡眠状态的工人执行Ticker::sleep。
    ///
    /// 如果ID在睡眠列表中：
    /// - 已存在，则更新线程恢复器，返回false。
    /// - 不存在，说明工人已被通知，返回true，并重新插入休眠列表。
    ///
    /// Returns `true` if the ticker was notified.
    fn update(&mut self, id: usize, waker: &Waker) -> bool {
        for item in &mut self.wakers {
            if item.0 == id {
                item.1.clone_from(waker);
                return false;
            }
        }

        self.wakers.push((id, waker.clone()));
        true
    }

    /// Removes a previously inserted sleeping ticker.
    /// 移除睡眠的工人，计数减一。
    ///
    /// 调用时机：
    /// - 当睡眠中的工人被通知，并成功获取到任务后，将自己变为清醒(Ticker::wake)时。
    ///
    /// 如果工人在睡眠列表中：
    /// - 已存在，ID放入复用列表，移除线程恢复器。返回false。
    /// - 不存在，返回true。(说明已被通知)。
    ///
    /// Returns `true` if the ticker was notified.
    fn remove(&mut self, id: usize) -> bool {
        self.count -= 1;
        self.free_ids.push(id);

        for i in (0..self.wakers.len()).rev() {
            if self.wakers[i].0 == id {
                self.wakers.remove(i);
                return false;
            }
        }
        true
    }

    /// Returns `true` if a sleeping ticker is notified or no tickers are sleeping.<br>
    ///
    /// 判断当前是否有工人在执行自我清醒流程：
    /// - 如果没有工人在执行清醒流程，count和wakers数量必然相等。(参见Sleepers的通知逻辑)
    /// - 如果有工人正在执行清醒流程，说明wakers中必然弹出了一个线程恢复器用于开启工人清醒流程，因此count和wakers数量不等。
    fn is_notified(&self) -> bool {
        self.count == 0 || self.count > self.wakers.len()
    }

    /// Returns notification waker for a sleeping ticker.
    ///
    /// If a ticker was notified already or there are no tickers, `None` will be returned.
    ///
    /// 通知一个睡眠中的工人执行自我清醒：
    /// - 如果count和wakers不等，说明当前正在有别的工人在执行自我唤醒流程，此操作在工人间互斥的。
    /// - 唤醒时，从休眠工人队列头部取一个线程恢复器，恢复工人所在线程，开启工人自我清醒流程。
    fn notify(&mut self) -> Option<Waker> {
        if self.wakers.len() == self.count {
            self.wakers.pop().map(|item| item.1)
        } else {
            None
        }
    }
}

/// Runs task one by one.
/// 工人状态小助手。
///
/// 两个职责：
/// - 根据任务获取逻辑，为工人获取任务(来执行)。
/// - 根据任务寻找结果，无任务时让工人进入休眠，有任务时唤醒工人。
struct Ticker<'a> {
    /// The executor state.
    /// 执行器引用。
    state: &'a State,

    /// Set to a non-zero sleeper ID when in sleeping state.
    /// 工人休眠时设为非0的休眠ID。
    ///
    /// States a ticker can be in:
    /// 1) Woken.
    /// 2a) Sleeping and unnotified.
    /// 2b) Sleeping and notified.
    ///
    /// 工人的状态可以是：
    /// 1) 0：清醒状态，正常处理任务。
    /// 2) n：休眠状态
    ///   2a) 休眠中且未被通知。
    ///   2b) 休眠中且已被通知，(马上就会被唤醒)
    sleeping: usize,
}

impl Ticker<'_> {
    /// Creates a ticker.
    /// 创建工人时钟，初始为清醒状态。
    fn new(state: &State) -> Ticker<'_> {
        Ticker { state, sleeping: 0 }
    }

    /// Moves the ticker into sleeping and unnotified state.
    ///
    /// Returns `false` if the ticker was already sleeping and unnotified.
    ///
    /// 将工人进入睡眠状态。
    ///
    /// 原状态如果是：
    /// - 清醒：将线程恢复器插入睡眠管理器Sleepers，回写睡眠ID。更新睡眠计数。
    /// - 睡眠：依据睡眠ID，更新或插入线程恢复器。不更新睡眠计数。
    ///
    /// 返回值：
    /// - 如果原先不是睡眠状态，或waker不存在，此时返回true。
    /// - 否则，工人已经是睡眠状态，且未被通知(waker存在)，则返回false。
    ///
    /// 更新工人自我唤醒互斥锁：
    /// - 依据当前是否有工人正在进行自我清醒流程，更新互斥锁。
    fn sleep(&mut self, waker: &Waker) -> bool {
        let mut sleepers = self.state.sleepers.lock().unwrap();

        match self.sleeping {
            // Move to sleeping state.
            0 => {
                self.sleeping = sleepers.insert(waker);
            }

            // Already sleeping, check if notified.
            id => {
                if !sleepers.update(id, waker) {
                    return false;
                }
            }
        }

        self.state
            .notified
            .store(sleepers.is_notified(), Ordering::Release);

        true
    }

    /// Moves the ticker into woken state.
    ///
    /// 将工人自我清醒。
    ///
    /// 依据当前状态：
    /// - 清醒：什么都不干。
    /// - 睡眠：从睡眠列表中移除线程恢复器，并更新互斥锁。
    fn wake(&mut self) {
        if self.sleeping != 0 {
            let mut sleepers = self.state.sleepers.lock().unwrap();
            sleepers.remove(self.sleeping);

            self.state
                .notified
                .store(sleepers.is_notified(), Ordering::Release);
        }
        self.sleeping = 0;
    }

    /// Waits for the next runnable task to run.
    /// 异步等待一个可用运行的任务，传入任务获取函数。（这里是从全局队列)
    async fn runnable(&mut self) -> Runnable {
        self.runnable_with(|| self.state.queue.pop().ok()).await
    }

    /// Waits for the next runnable task to run, given a function that searches for a task.
    /// 异步获取一个可执行的任务，通过提供的任务获取函数查找任务。
    ///
    /// 实现方式：
    /// - 将查找逻辑通过poll_fn附加上下文后转为一个Future，等待此Future完成。
    /// - 如果没查到，则将工人转入睡眠状态。如果之前已经是睡眠状态则返回Pending
    /// - 如果找到了，自身转为唤醒状态，然后传播唤醒一个其它睡眠中的工人，最终返回找到的任务。
    async fn runnable_with(&mut self, mut search: impl FnMut() -> Option<Runnable>) -> Runnable {
        future::poll_fn(|cx| {
            loop {
                match search() {
                    None => {
                        // Move to sleeping and unnotified state.
                        // 工人进入睡眠。
                        //
                        // 如果之前已经睡眠状态且count与waker一致，则返回false，让工人线程暂停。
                        if !self.sleep(cx.waker()) {
                            // If already sleeping and unnotified, return.
                            return Poll::Pending;
                        }
                    }
                    Some(r) => {
                        // Wake up.
                        self.wake();

                        // Notify another ticker now to pick up where this ticker left off, just in
                        // case running the task takes a long time.
                        // 传播唤醒另一个工人，防止当前工人执行太久。
                        self.state.notify();

                        return Poll::Ready(r);
                    }
                }
            }
        })
        .await
    }
}

impl Drop for Ticker<'_> {
    fn drop(&mut self) {
        // If this ticker is in sleeping state, it must be removed from the sleepers list.
        // 如果当前工人还处于睡眠状态，则先从睡眠列表中移除线程恢复器。
        if self.sleeping != 0 {
            // 移除线程恢复器。
            let mut sleepers = self.state.sleepers.lock().unwrap();
            let notified = sleepers.remove(self.sleeping);

            // 刷新工人清醒互斥锁。
            self.state
                .notified
                .store(sleepers.is_notified(), Ordering::Release);

            // If this ticker was notified, then notify another ticker.
            // 如果当前有工人正在进行清醒流程中，则先遗弃线程恢复器列表，再通知一次。
            if notified {
                drop(sleepers);
                self.state.notify();
            }
        }
    }
}

/// A worker in a work-stealing executor.
/// 工人，用在任务窃取型执行器中。
///
/// This is just a ticker that also has an associated local queue for improved cache locality.
struct Runner<'a> {
    /// The executor state.
    ///
    /// 执行器状态数据引用。
    state: &'a State,

    /// Inner ticker.
    /// 工人小工具。
    ticker: Ticker<'a>,

    /// The local queue.
    ///
    /// 工人本地队列。
    local: Arc<ConcurrentQueue<Runnable>>,

    /// Bumped every time a runnable task is found.
    ///
    /// 本工人已经poll过多少个任务，(本工人已经获取到多少个任务)。
    ticks: usize,
}

impl Runner<'_> {
    /// Creates a runner and registers it in the executor state.
    /// 创建一个工人，工人的本地队列会同时记录到执行器的本地队列列表中。
    fn new(state: &State) -> Runner<'_> {
        let runner = Runner {
            state,
            ticker: Ticker::new(state),
            local: Arc::new(ConcurrentQueue::bounded(512)),
            ticks: 0,
        };

        // 本地队列引用到执行器本地队列列表中。
        state
            .local_queues
            .write()
            .unwrap()
            .push(runner.local.clone());
        runner
    }

    /// Waits for the next runnable task to run.
    /// 异步获取下一个可执行任务。
    async fn runnable(&mut self, rng: &mut fastrand::Rng) -> Runnable {
        let runnable = self
            .ticker
            .runnable_with(|| {
                // Try the local queue.
                // 本地队列获取。
                if let Ok(r) = self.local.pop() {
                    return Some(r);
                }

                // Try stealing from the global queue.
                // 全局队列获取。
                if let Ok(r) = self.state.queue.pop() {
                    steal(&self.state.queue, &self.local);
                    return Some(r);
                }

                // Try stealing from other runners.
                // 从其它工人的队列窃取。
                let local_queues = self.state.local_queues.read().unwrap();

                // Pick a random starting point in the iterator list and rotate the list.
                // 以随机的位置旋转本地队列列表。目的是随机选一个队列作为首个被尝试窃取的队列。
                // - 生成随机数作为索引基点，start
                // - 序列复制一遍，追加到末尾，临时序列序列长度2n
                // - 基点之前的子序列抛弃，基点元素变为首个元素
                // - 从首个元素开始取n个元素。
                // - 最终序列从原始序列截取的内容：[start..n]+[0..start]
                //
                let n = local_queues.len();
                let start = rng.usize(..n);
                let iter = local_queues
                    .iter()
                    .chain(local_queues.iter())
                    .skip(start)
                    .take(n);

                // Remove this runner's local queue.
                // 过滤掉当前工人自己的队列。
                let iter = iter.filter(|local| !Arc::ptr_eq(local, &self.local));

                // Try stealing from each local queue in the list.
                // 按队列顺序尝试窃取任务，当遇到首个有任务的队列时，从中窃取一半任务，并返回窃取成功后的首个任务。
                for local in iter {
                    steal(local, &self.local);
                    if let Ok(r) = self.local.pop() {
                        return Some(r);
                    }
                }

                None
            })
            .await;

        // Bump the tick counter.
        // 本工人已经poll过了多少个任务。(也可以认为是，执行器已经成功获取到任务的次数)
        self.ticks = self.ticks.wrapping_add(1);

        // 每执行64次任务，就从全局队列窃取一组任务。
        // 窃取数量：(ls.len() + 1) / 2。
        if self.ticks % 64 == 0 {
            // Steal tasks from the global queue to ensure fair task scheduling.
            steal(&self.state.queue, &self.local);
        }

        runnable
    }
}

/// 遗弃工人
/// - 从执行器中移除当前工人的队列的引用。
/// - 将本地队列中的任务全部重新调度。
impl Drop for Runner<'_> {
    fn drop(&mut self) {
        // Remove the local queue.
        self.state
            .local_queues
            .write()
            .unwrap()
            .retain(|local| !Arc::ptr_eq(local, &self.local));

        // Re-schedule remaining tasks in the local queue.
        while let Ok(r) = self.local.pop() {
            r.schedule();
        }
    }
}

/// Steals some items from one queue into another.
/// 将任务从源队列移动到目标队列。移动数量为原队列任务的一半。
fn steal<T>(src: &ConcurrentQueue<T>, dest: &ConcurrentQueue<T>) {
    // Half of `src`'s length rounded up.
    let mut count = (src.len() + 1) / 2;

    if count > 0 {
        // Don't steal more than fits into the queue.
        if let Some(cap) = dest.capacity() {
            count = count.min(cap - dest.len());
        }

        // Steal tasks.
        for _ in 0..count {
            if let Ok(t) = src.pop() {
                assert!(dest.push(t).is_ok());
            } else {
                break;
            }
        }
    }
}

/// Debug implementation for `Executor` and `LocalExecutor`.
/// 调试执行器的实现。(判断是否初始化)
fn debug_executor(executor: &Executor<'_>, name: &str, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    // Get a reference to the state.
    // 获取状态引用。获取失败说明执行器未初始化
    let ptr = executor.state.load(Ordering::Acquire);
    if ptr.is_null() {
        // The executor has not been initialized.
        struct Uninitialized;

        impl fmt::Debug for Uninitialized {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("<uninitialized>")
            }
        }

        return f.debug_tuple(name).field(&Uninitialized).finish();
    }

    // SAFETY: If the state pointer is not null, it must have been
    // allocated properly by Arc::new and converted via Arc::into_raw
    // in state_ptr.
    let state = unsafe { &*ptr };

    debug_state(state, name, f)
}

/// Debug implementation for `Executor` and `LocalExecutor`.
/// 调试执行器的实现。(打印执行器状态内容)
fn debug_state(state: &State, name: &str, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    /// Debug wrapper for the number of active tasks.
    /// 激活任务。
    struct ActiveTasks<'a>(&'a Mutex<Slab<Waker>>);

    impl fmt::Debug for ActiveTasks<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self.0.try_lock() {
                Ok(lock) => fmt::Debug::fmt(&lock.len(), f),
                Err(TryLockError::WouldBlock) => f.write_str("<locked>"),
                Err(TryLockError::Poisoned(err)) => fmt::Debug::fmt(&err.into_inner().len(), f),
            }
        }
    }

    /// Debug wrapper for the local runners.
    /// 本地队列。
    struct LocalRunners<'a>(&'a RwLock<Vec<Arc<ConcurrentQueue<Runnable>>>>);

    impl fmt::Debug for LocalRunners<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self.0.try_read() {
                Ok(lock) => f
                    .debug_list()
                    .entries(lock.iter().map(|queue| queue.len()))
                    .finish(),
                Err(TryLockError::WouldBlock) => f.write_str("<locked>"),
                Err(TryLockError::Poisoned(_)) => f.write_str("<poisoned>"),
            }
        }
    }

    /// Debug wrapper for the sleepers.
    /// 任务搜索器的休闲管理器。
    struct SleepCount<'a>(&'a Mutex<Sleepers>);

    impl fmt::Debug for SleepCount<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self.0.try_lock() {
                Ok(lock) => fmt::Debug::fmt(&lock.count, f),
                Err(TryLockError::WouldBlock) => f.write_str("<locked>"),
                Err(TryLockError::Poisoned(_)) => f.write_str("<poisoned>"),
            }
        }
    }

    f.debug_struct(name)
        .field("active", &ActiveTasks(&state.active))
        .field("global_tasks", &state.queue.len())
        .field("local_runners", &LocalRunners(&state.local_queues))
        .field("sleepers", &SleepCount(&state.sleepers))
        .finish()
}

/// Runs a closure when dropped.
/// 资源守卫，可自定义收尾函数。
struct CallOnDrop<F: FnMut()>(F);

impl<F: FnMut()> Drop for CallOnDrop<F> {
    fn drop(&mut self) {
        (self.0)();
    }
}

pin_project! {
    /// A wrapper around a future, running a closure when dropped.
    /// 包装Future，Future被回收后，启用自定义收尾函数。
    struct AsyncCallOnDrop<Fut, Cleanup: FnMut()> {
        #[pin]
        future: Fut,
        cleanup: CallOnDrop<Cleanup>,
    }
}

impl<Fut, Cleanup: FnMut()> AsyncCallOnDrop<Fut, Cleanup> {
    fn new(future: Fut, cleanup: Cleanup) -> Self {
        Self {
            future,
            cleanup: CallOnDrop(cleanup),
        }
    }
}

impl<Fut: Future, Cleanup: FnMut()> Future for AsyncCallOnDrop<Fut, Cleanup> {
    type Output = Fut::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.project().future.poll(cx)
    }
}

/// 编译时检测一系列执行器相关类型的send和sync实现情况。
fn _ensure_send_and_sync() {
    use futures_lite::future::pending;

    fn is_send<T: Send>(_: T) {}
    fn is_sync<T: Sync>(_: T) {}
    fn is_static<T: 'static>(_: T) {}

    is_send::<Executor<'_>>(Executor::new());
    is_sync::<Executor<'_>>(Executor::new());

    let ex = Executor::new();
    is_send(ex.run(pending::<()>()));
    is_sync(ex.run(pending::<()>()));
    is_send(ex.tick());
    is_sync(ex.tick());
    is_send(ex.schedule());
    is_sync(ex.schedule());
    is_static(ex.schedule());

    /// ```compile_fail
    /// use async_executor::LocalExecutor;
    /// use futures_lite::future::pending;
    ///
    /// fn is_send<T: Send>(_: T) {}
    /// fn is_sync<T: Sync>(_: T) {}
    ///
    /// is_send::<LocalExecutor<'_>>(LocalExecutor::new());
    /// is_sync::<LocalExecutor<'_>>(LocalExecutor::new());
    ///
    /// let ex = LocalExecutor::new();
    /// is_send(ex.run(pending::<()>()));
    /// is_sync(ex.run(pending::<()>()));
    /// is_send(ex.tick());
    /// is_sync(ex.tick());
    /// ```
    fn _negative_test() {}
}
