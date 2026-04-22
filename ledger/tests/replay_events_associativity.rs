// This file is part of midnight-ledger.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//! Shows that `DustLocalState::replay_events` isn't associative: splitting
//! an event stream into chunks and folding chunk-by-chunk gives a different
//! `DustLocalState` than folding the whole stream in one call.
//!
//! This matters because two realistic callers disagree. A batch consumer
//! replays every event from genesis in one go. A streaming consumer (live
//! indexer, per-block materializer) calls `replay_events` once per block
//! or per event. For the same secret key and the same events, they should
//! land on the same state — but they don't.
//!
//! The split point that triggers it is a non-owner `DustInitialUtxo`
//! followed, later in the stream, by a `DustGenerationDtimeUpdate` for the
//! same generation. In the batch call, the collapse of the non-owner leaf
//! is deferred until after the whole iterator is drained, so the dtime
//! update runs against an uncollapsed leaf. In the split call, the first
//! chunk ends and its collapse fires immediately; when the dtime update
//! arrives in the next chunk, `update_from_evidence` re-fills the slot
//! that was just collapsed, and the state ends up bigger.
//!
//! The test asserts:
//!
//!     state.replay_events(sk, head ++ tail)
//!   ==
//!     state.replay_events(sk, head).replay_events(sk, tail)
//!
//! and currently fails.

use base_crypto::time::Timestamp;
use midnight_ledger::dust::{
    DustLocalState, DustPublicKey, DustSecretKey, INITIAL_DUST_PARAMETERS, InitialNonce,
};
use midnight_ledger::events::{Event, EventDetails};
use midnight_ledger::structure::{
    CNightGeneratesDustActionType, CNightGeneratesDustEvent, INITIAL_PARAMETERS, LedgerState,
    SystemTransaction,
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serialize::tagged_serialize;
use sha2::{Digest, Sha256};
use storage::db::InMemoryDB;

fn hash(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex::encode(h.finalize())
}

/// Build a `CNightGeneratesDustUpdate` system tx for a single event.
fn night_generates_dust_tx(
    action: CNightGeneratesDustActionType,
    owner: DustPublicKey,
    nonce: InitialNonce,
    time: Timestamp,
) -> SystemTransaction {
    SystemTransaction::CNightGeneratesDustUpdate {
        events: vec![CNightGeneratesDustEvent {
            action,
            nonce,
            owner,
            time,
            value: 10_000_000,
        }],
    }
}

/// Derive a fresh owner keypair from a seeded RNG.
fn derive_owner(rng: &mut StdRng) -> (DustSecretKey, DustPublicKey) {
    let sk = DustSecretKey::derive_secret_key(&rng.r#gen::<[u8; 32]>());
    let pk = DustPublicKey::from(sk.clone());
    (sk, pk)
}

/// Build a minimal event stream containing a non-owner `DustInitialUtxo`
/// followed by a later `DustGenerationDtimeUpdate` referencing the same
/// generation index. Any split point between the two triggers the bug.
fn emit_events() -> Vec<Event<InMemoryDB>> {
    let mut rng = StdRng::seed_from_u64(0x42);
    let (_owner_sk, owner_pk) = derive_owner(&mut rng);
    let nonce = InitialNonce(rng.r#gen());
    let t0 = Timestamp::from_secs(1_000);
    let t1 = t0 + INITIAL_DUST_PARAMETERS.time_to_cap();

    let ledger: LedgerState<InMemoryDB> = LedgerState::new("local-test");
    let create = night_generates_dust_tx(
        CNightGeneratesDustActionType::Create,
        owner_pk,
        nonce,
        t0,
    );
    let (ledger, ev_create) = ledger.apply_system_tx(&create, t0).expect("apply create");

    let destroy = night_generates_dust_tx(
        CNightGeneratesDustActionType::Destroy,
        owner_pk,
        nonce,
        t1,
    );
    let (_, ev_destroy) = ledger.apply_system_tx(&destroy, t1).expect("apply destroy");

    [ev_create, ev_destroy].concat()
}

/// Index of the first event whose content matches `pred`.
fn find_event<F>(events: &[Event<InMemoryDB>], pred: F, label: &str) -> usize
where
    F: Fn(&EventDetails<InMemoryDB>) -> bool,
{
    events
        .iter()
        .position(|e| pred(&e.content))
        .unwrap_or_else(|| panic!("test fixture must contain {label}"))
}

/// Locate the `DustInitialUtxo` and `DustGenerationDtimeUpdate` boundary in
/// the stream; panics if missing or out of order.
fn boundary(events: &[Event<InMemoryDB>]) -> (usize, usize) {
    let diu = find_event(
        events,
        |c| matches!(c, EventDetails::DustInitialUtxo { .. }),
        "DustInitialUtxo",
    );
    let dtime = find_event(
        events,
        |c| matches!(c, EventDetails::DustGenerationDtimeUpdate { .. }),
        "DustGenerationDtimeUpdate",
    );
    assert!(
        diu < dtime,
        "fixture ordering: DIU must precede dtime update; got diu={diu} dtime={dtime}"
    );
    (diu, dtime)
}

/// Viewer key unrelated to any owner in the stream — every DIU is a
/// non-owner DIU for this sk, which is the path that accumulates
/// `gen_collapses`.
fn viewer_sk() -> DustSecretKey {
    let mut rng = StdRng::seed_from_u64(0x1234);
    DustSecretKey::derive_secret_key(&rng.r#gen::<[u8; 32]>())
}

/// Fold the whole stream in a single `replay_events` call.
fn replay_batch(
    state0: &DustLocalState<InMemoryDB>,
    sk: &DustSecretKey,
    events: &[Event<InMemoryDB>],
) -> DustLocalState<InMemoryDB> {
    state0
        .replay_events(sk, events.iter())
        .expect("batch replay must succeed")
}

/// Fold the stream in two chunks, `head` then `tail`, split at `at`.
fn replay_split(
    state0: &DustLocalState<InMemoryDB>,
    sk: &DustSecretKey,
    events: &[Event<InMemoryDB>],
    at: usize,
) -> DustLocalState<InMemoryDB> {
    let (head, tail) = events.split_at(at);
    state0
        .replay_events(sk, head.iter())
        .expect("head replay must succeed")
        .replay_events(sk, tail.iter())
        .expect("tail replay must succeed")
}

/// Serialize to the wire-format bytes and log a labeled hash line.
fn dump(label: &str, state: &DustLocalState<InMemoryDB>) -> Vec<u8> {
    let mut bytes = Vec::new();
    tagged_serialize(state, &mut bytes).expect("serialize state");
    eprintln!("{label}: {:>6} B  sha256={}", bytes.len(), hash(&bytes));
    bytes
}

#[test]
fn replay_events_is_associative() {
    let events = emit_events();
    let (_, split_at) = boundary(&events);
    let sk = viewer_sk();
    let state0 = DustLocalState::<InMemoryDB>::new(INITIAL_PARAMETERS.dust);

    let batch = dump("batch", &replay_batch(&state0, &sk, &events));
    let split = dump("split", &replay_split(&state0, &sk, &events, split_at));

    assert_eq!(
        batch,
        split,
        "replay_events is not associative — splitting the event stream \
         between a non-owner DustInitialUtxo and its later \
         DustGenerationDtimeUpdate produces a different DustLocalState \
         than a single batch call. batch={} B, split={} B.",
        batch.len(),
        split.len(),
    );
}
