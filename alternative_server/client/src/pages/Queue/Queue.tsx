import moshiProcessorUrl from "../../audio-processor.ts?worker&url";
import { FC, useEffect, useState, useCallback, useRef, MutableRefObject } from "react";
import eruda from "eruda";
import { useSearchParams } from "react-router-dom";
import { Conversation } from "../Conversation/Conversation";
import { Button } from "../../components/Button/Button";
import { useModelParams } from "../Conversation/hooks/useModelParams";
import { env } from "../../env";
import { prewarmDecoderWorker } from "../../decoder/decoderWorker";

const LANGUAGE_OPTIONS = [
  { value: "auto", label: "Auto-detect" },
  { value: "vi", label: "Vietnamese (forced)" },
  { value: "en", label: "English (forced)" },
];

const ENGINE_OPTIONS = [
  { value: "piper", label: "Piper (Vietnamese support)" },
  { value: "kokoro", label: "Kokoro (English only)" },
];

const TEXT_PROMPT_PRESETS = [
  {
    label: "Assistant (Vietnamese)",
    text: "You are a helpful assistant with Vietnamese language support, that only speak in Vietnamese. Answer questions clearly and provide helpful information only in Vietnamese as appropriate.",
  },
  {
    label: "Customer Service",
    text: "You work for a company providing customer service. Be polite and helpful.",
  },
  {
    label: "Teacher",
    text: "You are a friendly teacher. Help students learn English through conversation and explanations.",
  }
];

interface HomepageProps {
  showMicrophoneAccessMessage: boolean;
  startConnection: () => Promise<void>;
  textPrompt: string;
  setTextPrompt: (value: string) => void;
  voicePrompt: string;
  setVoicePrompt: (value: string) => void;
  language: string;
  setLanguage: (value: string) => void;
  engine: string;
  setEngine: (value: string) => void;
}

const Homepage = ({
  startConnection,
  showMicrophoneAccessMessage,
  textPrompt,
  setTextPrompt,
  voicePrompt,
  setVoicePrompt,
  language,
  setLanguage,
  engine,
  setEngine,
}: HomepageProps) => {
  // Update voice options based on engine and language
  const VOICE_OPTIONS = engine === "kokoro" 
    ? ["af_bella", "af_sarah", "af_sky", "af_nicole", "am_adam", "am_michael", "bf_emma", "bf_isabella", "bm_george", "bm_lewis"]
    : language === "vi"
    ? ["vi_VN-vais1000-medium", "vi_VN-25hours_single-low", "vi_VN-vivos-x_low"]
    : ["en_US-lessac-medium", "en_US-amy-medium", "en_US-danny-medium", "en_GB-alba-medium", "en_GB-cori-medium"];
  
  // Auto-set language when engine changes
  const handleEngineChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newEngine = e.target.value;
    setEngine(newEngine);
    
    // Force English for Kokoro
    if (newEngine === "kokoro") {
      setLanguage("en");
      // Set default Kokoro voice if current is Piper voice
      if (!voicePrompt.startsWith("af_") && !voicePrompt.startsWith("am_") && !voicePrompt.startsWith("bf_") && !voicePrompt.startsWith("bm_")) {
        setVoicePrompt("af_bella");
      }
    }
  };
  
  // Handle language change for Piper
  const handleLanguageChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newLang = e.target.value;
    setLanguage(newLang);
    
    // Auto-set voice based on language for Piper
    if (engine === "piper") {
      if (newLang === "vi") {
        setVoicePrompt("vi_VN-vais1000-medium");
      } else if (newLang === "en") {
        setVoicePrompt("en_US-lessac-medium");
      }
    }
  };
  
  return (
    <div className="text-center h-screen w-screen p-4 flex flex-col items-center pt-8">
      <div className="mb-6">
        <h1 className="text-4xl text-black">Alternative Voice Stack</h1>
        <p className="text-sm text-gray-600 mt-2">
          Whisper + Piper + Ollama with Vietnamese support
        </p>
      </div>

      <div className="flex flex-grow justify-center items-center flex-col gap-6 w-full min-w-[500px] max-w-2xl">
        <div className="w-full">
          <label htmlFor="engine" className="block text-left text-base font-medium text-gray-700 mb-2">
            TTS Engine:
          </label>
          <select
            id="engine"
            name="engine"
            value={engine}
            onChange={handleEngineChange}
            className="w-full p-3 bg-white text-black border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-[#76b900] focus:border-transparent"
          >
            {ENGINE_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
          <p className="text-xs text-gray-500 mt-1">Kokoro: English only, Piper: Vietnamese support</p>
        </div>

        <div className="w-full">
          <label htmlFor="text-prompt" className="block text-left text-base font-medium text-gray-700 mb-2">
            Text Prompt:
          </label>
          <div className="border border-gray-300 rounded p-3 mb-3 bg-gray-50">
            <span className="text-xs font-medium text-gray-500 block mb-2">Examples:</span>
            <div className="flex flex-wrap gap-2 justify-center">
              {TEXT_PROMPT_PRESETS.map((preset) => (
                <button
                  key={preset.label}
                  onClick={() => setTextPrompt(preset.text)}
                  className="px-3 py-1 text-xs bg-white hover:bg-gray-100 text-gray-700 rounded-full border border-gray-300 transition-colors focus:outline-none focus:ring-2 focus:ring-[#76b900]"
                >
                  {preset.label}
                </button>
              ))}
            </div>
          </div>
          <textarea
            id="text-prompt"
            name="text-prompt"
            value={textPrompt}
            onChange={(e) => setTextPrompt(e.target.value)}
            className="w-full h-32 min-h-[80px] max-h-64 p-3 bg-white text-black border border-gray-300 rounded resize-y focus:outline-none focus:ring-2 focus:ring-[#76b900] focus:border-transparent"
            placeholder="Enter your text prompt..."
            maxLength={1000}
          />
          <div className="text-right text-xs text-gray-500 mt-1">
            {textPrompt.length}/1000
          </div>
        </div>

        <div className="w-full">
          <label htmlFor="language" className="block text-left text-base font-medium text-gray-700 mb-2">
            Language:
          </label>
          <select
            id="language"
            name="language"
            value={language}
            onChange={handleLanguageChange}
            disabled={engine === "kokoro"}
            className="w-full p-3 bg-white text-black border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-[#76b900] focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
          >
            {LANGUAGE_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
          <p className="text-xs text-gray-500 mt-1">
            {engine === "kokoro" ? "Kokoro only supports English" : "Force Vietnamese if auto-detection fails"}
          </p>
        </div>

        <div className="w-full">
          <label htmlFor="voice-prompt" className="block text-left text-base font-medium text-gray-700 mb-2">
            Voices:
          </label>
          <select
            id="voice-prompt"
            name="voice-prompt"
            value={voicePrompt}
            onChange={(e) => setVoicePrompt(e.target.value)}
            className="w-full p-3 bg-white text-black border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-[#76b900] focus:border-transparent"
          >
            {VOICE_OPTIONS.map((voice) => (
              <option key={voice} value={voice}>
                {voice}
              </option>
            ))}
          </select>
          <p className="text-xs text-gray-500 mt-1">
            {engine === "kokoro" ? "Kokoro English voices" : language === "vi" ? "Piper Vietnamese voices" : "Piper English voices"}
          </p>
        </div>

        {showMicrophoneAccessMessage && (
          <p className="text-center text-red-500">Please enable your microphone before proceeding</p>
        )}
        
        <Button onClick={async () => await startConnection()}>Connect</Button>
    </div>
    </div>
  );
}

export const Queue:FC = () => {
  const theme = "light" as const;  // Always use light theme
  const [searchParams] = useSearchParams();
  const overrideWorkerAddr = searchParams.get("worker_addr");
  const [hasMicrophoneAccess, setHasMicrophoneAccess] = useState<boolean>(false);
  const [showMicrophoneAccessMessage, setShowMicrophoneAccessMessage] = useState<boolean>(false);
  const modelParams = useModelParams();

  const audioContext = useRef<AudioContext | null>(null);
  const worklet = useRef<AudioWorkletNode | null>(null);
  
  // enable eruda in development
  useEffect(() => {
    if(env.VITE_ENV === "development") {
      eruda.init();
    }
    () => {
      if(env.VITE_ENV === "development") {
        eruda.destroy();
      }
    };
  }, []);

  const getMicrophoneAccess = useCallback(async () => {
    try {
      await window.navigator.mediaDevices.getUserMedia({ audio: true });
      setHasMicrophoneAccess(true);
      return true;
    } catch(e) {
      console.error(e);
      setShowMicrophoneAccessMessage(true);
      setHasMicrophoneAccess(false);
    }
    return false;
}, [setHasMicrophoneAccess, setShowMicrophoneAccessMessage]);

  const startProcessor = useCallback(async () => {
    if(!audioContext.current) {
      audioContext.current = new AudioContext();
      // Prewarm decoder worker as soon as we have audio context
      // This gives WASM time to load while user grants mic access
      prewarmDecoderWorker(audioContext.current.sampleRate);
    }
    if(worklet.current) {
      return;
    }
    let ctx = audioContext.current;
    ctx.resume();
    try {
      worklet.current = new AudioWorkletNode(ctx, 'moshi-processor');
    } catch (err) {
      await ctx.audioWorklet.addModule(moshiProcessorUrl);
      worklet.current = new AudioWorkletNode(ctx, 'moshi-processor');
    }
    worklet.current.connect(ctx.destination);
  }, [audioContext, worklet]);

  const startConnection = useCallback(async() => {
      await startProcessor();
      const hasAccess = await getMicrophoneAccess();
      if (hasAccess) {
      // Values are already set in modelParams, they get passed to Conversation
    }
  }, [startProcessor, getMicrophoneAccess]);

  return (
    <>
      {(hasMicrophoneAccess && audioContext.current && worklet.current) ? (
        <Conversation
        workerAddr={overrideWorkerAddr ?? ""}
        audioContext={audioContext as MutableRefObject<AudioContext|null>}
        worklet={worklet as MutableRefObject<AudioWorkletNode|null>}
        theme={theme}
        startConnection={startConnection}
        {...modelParams}
        />
      ) : (
        <Homepage
          startConnection={startConnection}
          showMicrophoneAccessMessage={showMicrophoneAccessMessage}
          textPrompt={modelParams.textPrompt}
          setTextPrompt={modelParams.setTextPrompt}
          voicePrompt={modelParams.voicePrompt}
          setVoicePrompt={modelParams.setVoicePrompt}
          language={modelParams.language}
          setLanguage={modelParams.setLanguage}
          engine={modelParams.engine}
          setEngine={modelParams.setEngine}
        />
      )}
    </>
  );
};
