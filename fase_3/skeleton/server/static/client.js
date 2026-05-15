const connectButton = document.getElementById('connect-button');
const followButton = document.getElementById('follow-button');

const audioOutButton = document.getElementById('audio-out-button');
const audioLeftButton = document.getElementById('audio-left-button');
const audioRightButton = document.getElementById('audio-right-button');
const audioMuteButton = document.getElementById('audio-mute-button');

let socket;

let audioPlayback = "out";
let audioContext;
let audioWorkletNode;

/////
// Socket connection and data handling
/////

function connectToServer() {
    socket = io("/frontend", { transports: ["websocket"] });

    socket.on('connect', () => {
        console.log('Connected to server');
    });

    socket.on('disconnect', () => {
        console.log('Connection closed');
    });


    socket.on('gsc_data', async (data) => {
        let timestamps = data['timestamps'];
        let gsc0 = data['gsc_left'];
        let gsc1 = data['gsc_right'];

        if (audioPlayback === "left") {
            await playAudio(gsc0);
        } else if (audioPlayback === "right") {
            await playAudio(gsc1);
        }

        for (let i = 0; i < gsc0.length; i++) {
            gsc_left.push({ x: timestamps[i], y: gsc0[i] });
            gsc_right.push({ x: timestamps[i], y: gsc1[i] });
        }
        gsc_left_chart.update();
        gsc_right_chart.update();

        doa_left.push({ x: timestamps[timestamps.length - 1], y: data['doa_left'] });
        doa_right.push({ x: timestamps[timestamps.length - 1], y: data['doa_right'] });
        doa_gt_0.push({ x: timestamps[timestamps.length - 1], y: data['doa_gt_0'] });
        doa_gt_1.push({ x: timestamps[timestamps.length - 1], y: data['doa_gt_1'] });
        doa_chart.update();

        snr_global.push({ x: timestamps[timestamps.length - 1], y: data['sir'] });
        snr_chart.update();
    });

    socket.on('out_data', async (data) => {
        let timestamps = data['timestamps'];
        let predicted_speaker = data['predicted_speaker'];
        let out = data['output_signal'];

        if (audioPlayback === "out") {
            await playAudio(out);
        }

        for (let i = 0; i < out.length; i++) {
            if (predicted_speaker === 0) {
                output_audio_0.push({ x: timestamps[i], y: out[i] });
            } else {
                output_audio_1.push({ x: timestamps[i], y: out[i] });
            }
        }

        out_chart.update();
    });

    socket.on('aad_data', (data) => {
        timestamps = data["timestamps"];
        attended_speaker = data["attended_speaker"];

        probablity_data.push({ x: timestamps[timestamps.length - 1], y: data["pred_prob"] });
        accuracy_data.push({ x: timestamps[timestamps.length - 1], y: data["accuracy"] });
        avg_accuracy_data.push({ x: timestamps[timestamps.length - 1], y: data["avg_accuracy"] });
        for (let i = 0; i < attended_speaker.length; i++) {
            attended_data.push({ x: timestamps[i], y: attended_speaker[i] });
        }
        attended_chart.update();
    });
}

/////
// TimeChart initialization
/////

const settings = {
    xScaleType: d3.scaleLinear,
    xRange: { min: -22 * 48000, max: 0 },
    realTime: true,
    zoom: {
        x: {
            autoRange: true,
            minDomainExtent: 50,
        },
        y: {
            autoRange: true,
            minDomainExtent: 1,
        }
    },
    tooltip: {
        enabled: true,
        xFormatter: (x) => new Date(x).toLocaleString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', fractionalSecondDigits: 3 }),
    },
    renderPaddingTop: 10,
    renderPaddingRight: 10,
    renderPaddingLeft: 45,
    renderPaddingBottom: 20,
}


const el_gsc_left_chart = document.getElementById('gsc_left');
let gsc_left = [];
const gsc_left_chart = new TimeChart(el_gsc_left_chart, {
    ...settings,
    series: [{
        name: "audio left",
        data: gsc_left,
        color: "#fcc621",
        lineType: 2,
    }],
    yRange: { min: -20000, max: 20000 },
});

const el_gsc_right_chart = document.getElementById('gsc_right');
let gsc_right = [];
const gsc_right_chart = new TimeChart(el_gsc_right_chart, {
    ...settings,
    series: [{
        name: "audio right",
        data: gsc_right,
        color: "#4287f5",
        lineType: 2,
    }],
    yRange: { min: -20000, max: 20000 },
});

const el_doa_chart = document.getElementById('doa');
let doa_left = [];
let doa_right = [];
let doa_gt_0 = [];
let doa_gt_1 = [];
const doa_chart = new TimeChart(el_doa_chart, {
    ...settings,
    series: [{
        name: "doa right",
        data: doa_right,
        color: "#4287f5",
        lineType: 2,
    }, {
        name: "doa left",
        data: doa_left,
        color: "#fcc621",
        lineType: 2,
    }, {
        data: doa_gt_0,
        color: "#adf25e",
        lineType: 2,
    }, {
        data: doa_gt_1,
        color: "#adf25e",
        lineType: 2,
    }],
    yRange: { min: -5, max: 185 },
});

const el_out_chart = document.getElementById('out');
let output_audio_0 = [];
let output_audio_1 = [];
const out_chart = new TimeChart(el_out_chart, {
    ...settings,
    series: [{
        name: "audio out (right)",
        data: output_audio_1,
        color: "#a974a6",
        lineType: 2,
    }, {
        name: "audio out (left)",
        data: output_audio_0,
        color: "#fe9146",
        lineType: 2,
    }],
    yRange: { min: -20000, max: 20000 },
});


const el_snr_chart = document.getElementById('snr');
let snr_global = [];
const snr_chart = new TimeChart(el_snr_chart, {
    ...settings,
    series: [{
        name: "sir",
        data: snr_global,
        color: "#ff6464",
        lineType: 2,
    }],
});

const el_attended_chart = document.getElementById('attended');
let attended_data = [];
let probablity_data = [];
let accuracy_data = [];
let avg_accuracy_data = [];
const attended_chart = new TimeChart(el_attended_chart, {
    ...settings,
    series: [{
        name: "probablity",
        data: probablity_data,
        color: "#ff204e",
        lineWidth: 3,
    }, {
        name: "accuracy (window)",
        data: accuracy_data,
        color: "#a0153e",
        lineType: 2,
    }, {
        name: "avg_accuracy",
        data: avg_accuracy_data,
        color: "#5d0e41",
        lineWidth: 2,
    }, {
        name: "attended speaker gt",
        data: attended_data,
        color: "#adf25e",
        lineWidth: 4,
    }],
    yRange: { min: 0, max: 1 },
});

/////
// Audio playback
/////

async function initAudio() {
    try {
        audioContext = new AudioContext({ sampleRate: 16000 });
        await audioContext.audioWorklet.addModule('static/pcm-processor.js');
        audioWorkletNode = new AudioWorkletNode(audioContext, 'pcm-processor');
        audioWorkletNode.connect(audioContext.destination);
    } catch (error) {
        console.error('Error initializing audio:', error);
    }
}

async function playAudio(data) {
    if (!audioContext || !audioWorkletNode) {
        return;
    }

    try {
        await audioWorkletNode.port.postMessage({ message: 'audioData', audioData: data });
    } catch (error) {
        console.error('Error playing audio:', error);
    }
}

/////
// EventListeners
/////

document.addEventListener('DOMContentLoaded', () => {
    connectToServer();
});

connectButton.addEventListener('click', () => {
    initAudio();
});

followButton.addEventListener('click', function () {
    gsc_left_chart.options.realTime = true;
    gsc_right_chart.options.realTime = true;
    doa_chart.options.realTime = true;
    out_chart.options.realTime = true;
    snr_chart.options.realTime = true;
    attended_chart.options.realTime = true;
});

audioOutButton.addEventListener('click', function () {
    audioPlayback = "out";
});

audioLeftButton.addEventListener('click', function () {
    audioPlayback = "left";
});

audioRightButton.addEventListener('click', function () {
    audioPlayback = "right";
});

audioMuteButton.addEventListener('click', function () {
    audioPlayback = "mute";
});
