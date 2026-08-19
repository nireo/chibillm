#pragma once

// Ideas for the scheduler:
// - The batch of requests must be dynamic -> having two request batch where one finishes
//   then the scheduler should fetch new requests there.
// - Prefill should be scheduled separately since they are different
