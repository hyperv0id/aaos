# Final fix report

## Finding

The `wait_for_idle_multiple_waiters` acceptance test serialized the runner and
waiters behind one `tokio::sync::Mutex<Agent>`. Each waiter acquired the lock
only after `prompt()` had already finished, so the test never exercised two
waiters sharing the active run's idle watch barrier.

## Fix

Added `wait_for_idle_multiple_waiters_share_active_barrier` in the
`agent.rs` unit-test module. It constructs an `ActiveRun` directly with a
`watch::channel(false)`, obtains two `Agent::wait_for_idle()` futures while the
shared idle value is false, asserts neither future resolves during a bounded
poll, sends `true` through the shared sender, and then asserts both futures
resolve.

Adjusted the acceptance-test comment to state that the shared mutex serializes
the runner and waiters, and that the new unit test is the focused coverage for
concurrent waiters on the same active idle barrier.

## Status

Test-only change. Production behavior and public APIs are unchanged.

## Fix round 2

The first unit-test commit missed that `tokio::select!` mutably borrows each
wait future. Marked `wait1` and `wait2` as `mut` so the bounded pending check
and the subsequent awaits compile.