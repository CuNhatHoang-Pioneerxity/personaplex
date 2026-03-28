import { useCallback, useRef, useState } from "react";
import Recorder from "opus-recorder";
import encoderPath from "opus-recorder/dist/encoderWorker.min.js?url";
import { useMediaContext } from "../MediaContext";

export enum UserMediaStatuses {
  IDLE = "IDLE",
  READY = "READY",
  WAITING_FOR_PERMISSION = "WAITING_FOR_PERMISSION",
  ERROR = "ERROR",
  RECORDING = "RECORDING",
  STOPPED = "STOPPED",
  STOPPING = "STOPPING",
}

type useUserAudioArgs = {
  constraints: MediaStreamConstraints;
  onDataChunk?: (chunk: Uint8Array) => void;
  onRecordingStart?: () => void;
  onRecordingStop?: () => void;
};

export const useUserAudio = ({
  constraints,
  onDataChunk,
  onRecordingStart = () => {},
  onRecordingStop = () => {},
}: useUserAudioArgs) => {
  const { stereoMerger, audioContext, micDuration } = useMediaContext();
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<UserMediaStatuses>(
    UserMediaStatuses.IDLE,
  );

  const recorder = useRef<any>(null);

  const getMediaStream = useCallback(async () => {
    setStatus(UserMediaStatuses.WAITING_FOR_PERMISSION);
    try {
      const stream =
        await window.navigator.mediaDevices.getUserMedia(constraints);
      setStatus(UserMediaStatuses.IDLE);
      return stream;
    } catch (error: any) {
      console.error(error);
      setError(error.name);
      setStatus(UserMediaStatuses.ERROR);
      return null;
    }
  }, [constraints, setStatus]);

  const startRecordingUser = useCallback(async () => {
    console.log(Date.now() % 1000, "Starting recording in user audio");
    const mediaStream = await getMediaStream();
    if (mediaStream) {
      const analyser = audioContext.current.createAnalyser();
      const source = audioContext.current.createMediaStreamSource(mediaStream);
      source.connect(analyser);
      source.connect(stereoMerger.current, 0, 1);

      const recorderOptions = {
        mediaTrackConstraints: constraints,
        encoderPath,
        bufferLength: Math.round(960 * audioContext.current.sampleRate / 24000),
        encoderFrameSize: 20,
        encoderSampleRate: 24000,
        maxFramesPerPage: 2,
        numberOfChannels: 1,
        recordingGain: 1,
        resampleQuality: 3,
        encoderComplexity: 0,
        encoderApplication: 2049,
        streamPages: true,
      };
      let chunk_idx = 0;
      let lastpos = 0;
      recorder.current = new Recorder(recorderOptions);
      recorder.current.ondataavailable = (data: Uint8Array) => {
        micDuration.current = recorder.current.encodedSamplePosition / 48000;
        if (chunk_idx < 5) {
          console.log(Date.now() % 1000, "Mic Data chunk", chunk_idx++, (recorder.current.encodedSamplePosition - lastpos) / 48000, micDuration.current);
          lastpos = recorder.current.encodedSamplePosition;
        }
        if (onDataChunk) {
          onDataChunk(data);
        }
      };
      recorder.current.onstart = () => {
        setStatus(UserMediaStatuses.RECORDING);
        onRecordingStart();
      };
      recorder.current.onstop = () => {
        setStatus(UserMediaStatuses.STOPPED);
        source.disconnect();
        onRecordingStop();
        recorder.current = null;
      };

      if (recorder.current) {
        recorder.current.start();
      }

      return {
        analyser,
        mediaStream,
        source,
      };
    }
    return {
      analyser: null,
      mediaStream: null,
      source: null,
    };
  }, [setStatus, onDataChunk, onRecordingStart, onRecordingStop]);

  const stopRecording = useCallback(() => {
    setStatus(UserMediaStatuses.STOPPING);
    if (recorder.current) {
      recorder.current.stop();
    }
  }, [setStatus]);

  return {
    status,
    error,
    startRecordingUser,
    stopRecording,
  };
};
