'use strict';

// The diagnostics loader supplies these addresses only for the verified build.
// Gadget runs this script in the VIctory process before the title starts.
rpc.exports = {
    init(_stage, parameters) {
        const ready = new NativeFunction(ptr(parameters.ready), 'void', ['int']);
        try {
            const image = ptr(parameters.image);
            const handler = image.add(0x1d65f10);
            const registrar = image.add(0x1bb1cf0);
            const record = new NativeFunction(ptr(parameters.record), 'uint64',
                ['uint32', 'uint32', 'pointer', 'uint64']);

            Interceptor.attach(registrar, {
                onEnter(args) {
                    if (args[0].equals(handler) && args[1].toUInt32() === 0x9a0892b1) {
                        record(0, 0, this.returnAddress, 0);
                    }
                }
            });
            Interceptor.attach(handler, {
                onEnter(args) {
                    const subcode = args[0].toUInt32();
                    const entry = record(1, subcode, this.returnAddress, 0);
                    // Gum keeps this object per invocation, including nested calls.
                    if (subcode === 0x20) {
                        this.activationEntry = entry;
                    }
                },
                onLeave() {
                    if (this.activationEntry !== undefined) {
                        record(2, 0x20, this.returnAddress, this.activationEntry);
                    }
                }
            });
            Interceptor.flush();
        } catch (error) {
            console.error('activation trace could not arm: ' + error);
            ready(0);
            return;
        }
        ready(1);
    }
};
