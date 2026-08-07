//! Bounds how long an accepted connection may go without producing a request.
//!
//! Hyper's HTTP/1 header timeout starts after HTTP version detection, while HTTP/2
//! has no equivalent header timeout. The stream therefore keeps a hard deadline
//! through protocol detection, disarms it while requests are active, and rearms it
//! when the last service future completes. Transport-level traffic such as HTTP/2
//! PING frames does not extend the deadline.

use std::{
    future::{ready, Future, Ready},
    io::{self, IoSlice},
    pin::Pin,
    sync::{Arc, Mutex, MutexGuard},
    task::{Context, Poll, Waker},
    time::Duration,
};

use axum_server::accept::Accept;
use http::Request;
use tokio::{
    io::{AsyncRead, AsyncWrite, ReadBuf},
    net::TcpStream,
    time::{sleep_until, Instant, Sleep},
};
use tower_service::Service;

#[derive(Clone, Copy, Debug)]
pub(crate) struct RequestDeadlineAcceptor {
    timeout: Duration,
}

impl RequestDeadlineAcceptor {
    pub(crate) fn new(timeout: Duration) -> Self {
        Self { timeout }
    }
}

impl<S> Accept<TcpStream, S> for RequestDeadlineAcceptor {
    type Stream = RequestDeadlineStream;
    type Service = RequestDeadlineService<S>;
    type Future = Ready<io::Result<(Self::Stream, Self::Service)>>;

    fn accept(&self, stream: TcpStream, service: S) -> Self::Future {
        let deadline = Instant::now() + self.timeout;
        let state = Arc::new(RequestDeadlineState::new(self.timeout, deadline));

        ready(Ok((
            RequestDeadlineStream {
                stream,
                deadline_sleep: Box::pin(sleep_until(deadline)),
                state: Arc::clone(&state),
            },
            RequestDeadlineService { service, state },
        )))
    }
}

#[derive(Debug)]
struct RequestDeadlineState {
    inner: Mutex<RequestDeadlineStateInner>,
    timeout: Duration,
}

#[derive(Debug)]
struct RequestDeadlineStateInner {
    active_requests: usize,
    deadline: Option<Instant>,
    transport_waker: Option<Waker>,
}

impl RequestDeadlineState {
    fn new(timeout: Duration, deadline: Instant) -> Self {
        Self {
            inner: Mutex::new(RequestDeadlineStateInner {
                active_requests: 0,
                deadline: Some(deadline),
                transport_waker: None,
            }),
            timeout,
        }
    }

    fn poll_deadline(&self, cx: &Context<'_>) -> Option<Instant> {
        let mut inner = self.lock_inner();
        if inner
            .transport_waker
            .as_ref()
            .is_none_or(|waker| !waker.will_wake(cx.waker()))
        {
            inner.transport_waker = Some(cx.waker().clone());
        }

        inner.deadline
    }

    fn start_request(&self) {
        let waker = {
            let mut inner = self.lock_inner();
            let was_idle = inner.active_requests == 0;
            inner.active_requests += 1;

            if was_idle {
                inner.deadline = None;
                inner.transport_waker.take()
            } else {
                None
            }
        };

        if let Some(waker) = waker {
            waker.wake();
        }
    }

    fn finish_request(&self) {
        let waker = {
            let mut inner = self.lock_inner();
            debug_assert!(
                inner.active_requests > 0,
                "a finished request must have been active"
            );
            if inner.active_requests == 0 {
                return;
            }

            inner.active_requests -= 1;
            if inner.active_requests == 0 {
                inner.deadline = Some(Instant::now() + self.timeout);
                inner.transport_waker.take()
            } else {
                None
            }
        };

        if let Some(waker) = waker {
            waker.wake();
        }
    }

    fn lock_inner(&self) -> MutexGuard<'_, RequestDeadlineStateInner> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

pub(crate) struct RequestDeadlineStream {
    stream: TcpStream,
    deadline_sleep: Pin<Box<Sleep>>,
    state: Arc<RequestDeadlineState>,
}

impl AsyncRead for RequestDeadlineStream {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        if let Some(deadline) = self.state.poll_deadline(cx) {
            if deadline != self.deadline_sleep.deadline() {
                self.deadline_sleep.as_mut().reset(deadline);
            }

            if self.deadline_sleep.as_mut().poll(cx).is_ready() {
                return Poll::Ready(Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "connection received no HTTP request before deadline",
                )));
            }
        }

        Pin::new(&mut self.stream).poll_read(cx, buffer)
    }
}

impl AsyncWrite for RequestDeadlineStream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buffer: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.stream).poll_write(cx, buffer)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.stream).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.stream).poll_shutdown(cx)
    }

    fn is_write_vectored(&self) -> bool {
        self.stream.is_write_vectored()
    }

    fn poll_write_vectored(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buffers: &[IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.stream).poll_write_vectored(cx, buffers)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RequestDeadlineService<S> {
    service: S,
    state: Arc<RequestDeadlineState>,
}

pin_project_lite::pin_project! {
    pub(crate) struct RequestDeadlineFuture<F> {
        #[pin]
        future: F,
        request_guard: Option<ActiveRequestGuard>,
    }
}

impl<F: Future> Future for RequestDeadlineFuture<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut this = self.project();
        let output = std::task::ready!(this.future.as_mut().poll(cx));
        this.request_guard.take();
        Poll::Ready(output)
    }
}

struct ActiveRequestGuard {
    state: Arc<RequestDeadlineState>,
}

impl ActiveRequestGuard {
    fn new(state: Arc<RequestDeadlineState>) -> Self {
        state.start_request();
        Self { state }
    }
}

impl Drop for ActiveRequestGuard {
    fn drop(&mut self) {
        self.state.finish_request();
    }
}

impl<S, B> Service<Request<B>> for RequestDeadlineService<S>
where
    S: Service<Request<B>>,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = RequestDeadlineFuture<S::Future>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.service.poll_ready(cx)
    }

    fn call(&mut self, request: Request<B>) -> Self::Future {
        let request_guard = ActiveRequestGuard::new(Arc::clone(&self.state));
        RequestDeadlineFuture {
            future: self.service.call(request),
            request_guard: Some(request_guard),
        }
    }
}

#[cfg(test)]
mod tests;
