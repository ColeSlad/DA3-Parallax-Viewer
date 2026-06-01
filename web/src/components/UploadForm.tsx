import { useRef, useState, type DragEvent, type ChangeEvent } from 'react'

const MAX_IMAGES = 30
const ALLOWED_MIME = new Set(['image/jpeg', 'image/png', 'image/webp'])

interface Props {
  onSubmit: (files: File[]) => void
  disabled: boolean
}

export function UploadForm({ onSubmit, disabled }: Props) {
  const [files, setFiles] = useState<File[]>([])
  const [error, setError] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  function validate(incoming: File[]): string | null {
    if (incoming.length === 0) return 'Select at least one image.'
    if (incoming.length > MAX_IMAGES) return `Too many images (max ${MAX_IMAGES}).`
    for (const f of incoming) {
      if (!ALLOWED_MIME.has(f.type)) return `${f.name}: unsupported type. Use JPEG, PNG, or WebP.`
    }
    return null
  }

  function apply(incoming: File[]) {
    const err = validate(incoming)
    setError(err)
    if (!err) setFiles(incoming)
    else setFiles([])
  }

  function onChange(e: ChangeEvent<HTMLInputElement>) {
    apply(Array.from(e.target.files ?? []))
  }

  function onDrop(e: DragEvent) {
    e.preventDefault()
    setDragging(false)
    apply(Array.from(e.dataTransfer.files))
  }

  function handleSubmit() {
    const err = validate(files)
    if (err) { setError(err); return }
    onSubmit(files)
  }

  return (
    <div style={{ maxWidth: 560, margin: '80px auto', fontFamily: 'system-ui, sans-serif' }}>
      <h1 style={{ marginBottom: 24 }}>3D Reconstruction</h1>

      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        style={{
          border: `2px dashed ${dragging ? '#2563eb' : '#cbd5e1'}`,
          borderRadius: 8,
          padding: 32,
          textAlign: 'center',
          cursor: 'pointer',
          background: dragging ? '#eff6ff' : '#f8fafc',
          transition: 'all 0.15s',
        }}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept="image/jpeg,image/png,image/webp"
          style={{ display: 'none' }}
          onChange={onChange}
        />
        <p style={{ margin: 0, color: '#64748b' }}>
          {files.length > 0
            ? `${files.length} image${files.length !== 1 ? 's' : ''} selected`
            : 'Drop images here or click to select'}
        </p>
        <p style={{ margin: '4px 0 0', fontSize: 12, color: '#94a3b8' }}>
          JPEG · PNG · WebP · max {MAX_IMAGES} files
        </p>
      </div>

      {error && (
        <p style={{ color: '#dc2626', marginTop: 8, fontSize: 14 }}>{error}</p>
      )}

      {files.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 12 }}>
          {files.map((f) => (
            <img
              key={f.name}
              src={URL.createObjectURL(f)}
              alt={f.name}
              style={{ width: 64, height: 64, objectFit: 'cover', borderRadius: 4 }}
            />
          ))}
        </div>
      )}

      <button
        onClick={handleSubmit}
        disabled={disabled || files.length === 0}
        style={{
          marginTop: 20,
          padding: '10px 24px',
          background: '#2563eb',
          color: '#fff',
          border: 'none',
          borderRadius: 6,
          fontSize: 15,
          cursor: files.length === 0 || disabled ? 'not-allowed' : 'pointer',
          opacity: files.length === 0 || disabled ? 0.5 : 1,
        }}
      >
        Reconstruct
      </button>
    </div>
  )
}
