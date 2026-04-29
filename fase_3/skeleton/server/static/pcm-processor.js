class PCMProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.buffer = new Float32Array();

        this.port.onmessage = (event) => {
            let newBuffer = new Float32Array(this.buffer.length + event.data.audioData.length);

            newBuffer.set(this.buffer, 0);
            for (let i = this.buffer.length, j = 0; j < event.data.audioData.length; i++, j++) {
                var int = event.data.audioData[j];
                var float = (int >= 0x8000) ? -(0x10000 - int) / 0x8000 : int / 0x7FFF;
                newBuffer[i] = float;
            }

            this.buffer = newBuffer;
        };
    }

    process(inputs, outputs, parameters) {
        const output = outputs[0];
        const channel = output[0];

        const bufferLength = Math.min(channel.length, this.buffer.length);
        channel.set(this.buffer.subarray(0, bufferLength));
        this.buffer = this.buffer.subarray(bufferLength);
        return true;
    }
}

registerProcessor('pcm-processor', PCMProcessor);
