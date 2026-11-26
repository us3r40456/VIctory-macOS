#define RS "RootConstants(num32BitConstants=4, b0), SRV(t0), UAV(u1), UAV(u0), SRV(t1)"
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint counted; };
// The title's records, read only; the clamped ones go to the list's own.
ByteAddressBuffer args : register(t0);
RWByteAddressBuffer dest : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
// The title's record count: read only if `counted`.
ByteAddressBuffer countWord : register(t1);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
// Clamps the record at `at` into `dest`; its cost.
uint clamped(uint at) {
    uint cost;
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint count = min(args.Load(at), 1u << 24);
        uint instances = min(args.Load(at + 4), (1u << 26) / max(count, 1u));
        cost = count * instances;
        dest.Store2(at, uint2(count, instances));
        // The rest of the record as it is: first vertex and instance (4 words
        // in all), or first index, base vertex and first instance (5).
        if (kind == 1) {
            dest.Store3(at + 8, args.Load3(at + 8));
        } else {
            dest.Store2(at + 8, args.Load2(at + 8));
        }
    } else if (kind == 2) {
        uint3 g = min(args.Load3(at), uint3(65535, 65535, 65535));
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64;
        dest.Store3(at, g);
    } else {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = args.Load3(at);
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128;
        dest.Store3(at, g);
    }
    return cost;
}
// A record past the budget (no cost reaches it).
static const uint PAST = 0xffffffffu;
groupshared uint costs[64];
// One group walks the records in tiles of 64: every lane clamps its own
// record, then the first lane charges the tile's records in order. No
// other group shares the budget: a shared add wraps once enough records
// are rejected, and a compare-exchange retries more the more records fit.
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint lane : SV_GroupIndex) {
    uint left = 0;
    if (lane == 0) {
        left = BUDGET - min(budget.Load(0), BUDGET);
    }
    // ExecuteIndirect runs min(max, count) records: no others are clamped or
    // charged. The loop still counts tiles up to `records`, a constant, as
    // the compiler cannot prove that a value read from memory is the same
    // in every lane, and barriers need that; tiles past `walked` only wait
    // at them.
    uint walked = records;
    if (counted != 0) {
        walked = min(records, countWord.Load(0));
    }
    for (uint tile = 0; tile < records / 64 + min(records % 64, 1u); tile++) {
        uint record = tile * 64 + lane;
        uint at = record * stride;
        uint cost = 0;
        if (record < walked) {
            cost = clamped(at);
        }
        costs[lane] = cost;
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0 && tile * 64 < walked) {
            for (uint i = 0; i < 64; i++) {
                if (costs[i] > left) {
                    // The first record that does not fit takes what is
                    // left, so those after it run empty too.
                    costs[i] = PAST;
                    left = 0;
                } else {
                    left -= costs[i];
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (costs[lane] == PAST) {
            // Draws keep their layout with no vertices; groups become zero.
            dest.Store(at, 0);
        }
    }
    if (lane == 0) {
        budget.Store(0, BUDGET - left);
    }
}
