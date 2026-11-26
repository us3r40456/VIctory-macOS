'use strict';

// Local read-only observations at the Series 1.0.2.1 initialization checks.
rpc.exports = {
    init(_stage, parameters) {
        const ready = new NativeFunction(ptr(parameters.ready), 'void', ['int']);
        const listeners = [];
        try {
            const image = ptr(parameters.image);
            const record = new NativeFunction(ptr(parameters.record), 'void',
                ['uint32', 'uint32', 'pointer', 'pointer', 'pointer', 'pointer']);
            function bytes(at, n) {
                return Array.from(new Uint8Array(at.readByteArray(n)));
            }
            function verify(at, expected) {
                if (bytes(at, expected.length).some((v, i) => v !== expected[i])) {
                    throw new Error('unexpected bytes at image+' + at.sub(image));
                }
            }
            const sites = [
                [0x63db9, [0x0f, 0xb7, 0x83, 0xe4, 0, 0, 0]], // load node word
                [0x63dd1, [0x8a, 0x83, 0xe1, 0, 0, 0]],       // load node byte
                [0x63de8, [0x48, 0x8b, 0x83, 0xe8, 0, 0, 0]], // load optional object
                [0x63e06, []],          // prepare optional call
                [0x63e14, [0xc6, 0x83, 0xf7, 0, 0, 0, 1]],    // after optional return
                [0x63e23, []],          // enter accepted path
                [0x63edd, [0xc7, 0x83, 0xf0, 0, 0, 0, 3, 0, 0, 0]] // write state 3
            ];
            verify(image.add(0x63dc0), [0x66, 0x85, 0xc0, 0x74, 0x0c]);
            verify(image.add(0x63dcb), [0x0f, 0x84, 0x0c, 1, 0, 0]);
            verify(image.add(0x63de2), [0x0f, 0x85, 0xf5, 0, 0, 0]);
            verify(image.add(0x63e1b), [0x84, 0xc0, 0x0f, 0x84, 0xba, 0, 0, 0]);
            // RAX at the optional-call preparation must still be the object
            // used by the actual indirect call. Verify this on the loaded image.
            const afterCall = image.add(0x63e14);
            let call;
            for (let p = image.add(0x63e06); p.compare(afterCall) < 0; p = call.next) {
                call = Instruction.parse(p);
                if (call.mnemonic === 'call') break;
                if (!['and', 'lea', 'mov', 'xor'].includes(call.mnemonic) ||
                    /^(?:rax|eax|ax|al|ah),/i.test(call.opStr)) {
                    throw new Error('unexpected optional call preparation at image+' + p.sub(image));
                }
            }
            if (!call || call.mnemonic !== 'call' || !call.next.equals(afterCall) ||
                !/\[rax\s*\+\s*(?:0x50|80)\]/i.test(call.opStr)) {
                throw new Error('unexpected optional callback before image+0x63e14');
            }

            // Keep branches and calls outside the relocated instruction window.
            const windows = sites.map(([rva, expected], i) => {
                const at = image.add(rva);
                let end = at.add(expected.length);
                if (expected.length !== 0) {
                    verify(at, expected);
                } else {
                    do {
                        const ins = Instruction.parse(end);
                        if (!['and', 'mov', 'lea', 'xor'].includes(ins.mnemonic)) {
                            throw new Error('unexpected successor at image+' + end.sub(image));
                        }
                        end = ins.next;
                    } while (end.sub(at).toUInt32() < 5);
                }
                const gap = i + 1 === sites.length ? 8 : sites[i + 1][0] - end.sub(image).toUInt32();
                if (gap <= 0) throw new Error('overlapping trace sites');
                return {at, end, tail: bytes(end, Math.min(8, gap))};
            });
            // Detect branches that enter the middle of a prospective probe window.
            for (let p = image.add(0x63cd0); p.compare(image.add(0x64009)) < 0;) {
                const ins = Instruction.parse(p);
                if (/^(?:j|loop|call)/.test(ins.mnemonic) && /^0x[0-9a-f]+$/i.test(ins.opStr)) {
                    const dst = ptr(ins.opStr);
                    if (windows.some(w => dst.compare(w.at) > 0 && dst.compare(w.end) < 0)) {
                        throw new Error('branch enters trace window at image+' + dst.sub(image));
                    }
                }
                p = ins.next;
            }
            windows.forEach(({at}, index) => {
                listeners.push(Interceptor.attach(at, function () {
                    const c = this.context;
                    record(index, this.threadId, c.rsp, c.rbx, c.r12, c.rax);
                }));
            });
            Interceptor.flush();
            windows.forEach(w => verify(w.end, w.tail));
        } catch (error) {
            for (const listener of listeners) listener.detach();
            Interceptor.flush();
            console.error('init gate trace could not arm: ' + error);
            ready(0);
            return;
        }
        ready(1);
    }
};
