/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_QUEUE_API_PATH: string
  readonly VITE_QUEUE_API_URL?: string
  readonly DEV?: boolean
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
