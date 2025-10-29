import { useState, useRef, useEffect } from 'react'
import './InputArea.css'

interface InputAreaProps {
  onSendMessage: (message: string) => void
  isProcessing: boolean
  onReset?: () => void
  exampleText?: string
  onFileUpload?: (file: File) => void
  uploadedFile?: File | null
}

const InputArea = ({ onSendMessage, isProcessing, exampleText, onFileUpload, uploadedFile }: InputAreaProps) => {
  const [input, setInput] = useState('')
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [showImageModal, setShowImageModal] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  
  // Update preview when uploadedFile changes
  useEffect(() => {
    if (uploadedFile && uploadedFile.type.startsWith('image/')) {
      const reader = new FileReader()
      reader.onloadend = () => {
        setImagePreview(reader.result as string)
      }
      reader.readAsDataURL(uploadedFile)
    } else if (!uploadedFile) {
      setImagePreview(null)
      setShowImageModal(false)
    }
  }, [uploadedFile])

  // Handle example text population
  useEffect(() => {
    if (exampleText) {
      setInput(exampleText)
      // Auto-resize textarea when example is loaded
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto'
        textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 120) + 'px'
      }
    }
  }, [exampleText])
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (input.trim() && !isProcessing) {
      onSendMessage(input.trim())
      setInput('')
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto'
      }
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)
    // Auto-resize textarea
    e.target.style.height = 'auto'
    e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px'
  }

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault()
    e.stopPropagation()
    const file = e.target.files?.[0]
    if (file && onFileUpload) {
      // Call the upload handler - preview will be updated via useEffect
      onFileUpload(file)
    }
    // Reset the input so the same file can be selected again if needed
    if (e.target) {
      e.target.value = ''
    }
  }

  const handleUploadClick = () => {
    fileInputRef.current?.click()
  }

  return (
    <>
      {/* Image Preview Modal - Outside form for proper positioning */}
      {showImageModal && imagePreview && (
        <div className="image-modal-overlay" onClick={() => setShowImageModal(false)}>
          <div className="image-modal-content" onClick={(e) => e.stopPropagation()}>
            <img src={imagePreview} alt="Preview" />
            <button 
              className="image-modal-close"
              onClick={() => setShowImageModal(false)}
            >
              ✕
            </button>
          </div>
        </div>
      )}
      
      <div className="input-area">
        <form onSubmit={handleSubmit} className="input-container">
          {/* Hidden File Input */}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*,.pdf,.txt,.doc,.docx"
            onChange={handleFileUpload}
            style={{ display: 'none' }}
          />
          
          {/* Upload Button / Image Preview */}
          {imagePreview ? (
            <div 
              className="upload-btn image-preview"
              title="Click to preview image"
              onClick={() => setShowImageModal(true)}
            >
              <img src={imagePreview} alt="Preview" />
            </div>
          ) : (
          <button
            type="button"
            className="upload-btn"
            title="Upload Image or Document"
            onClick={handleUploadClick}
          >
            <svg viewBox="0 0 24 24">
              <path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z"/>
            </svg>
          </button>
        )}
        
        <textarea
          ref={textareaRef}
          value={input}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          placeholder="Enter your premises and conclusion here..."
          className="input-field"
          disabled={isProcessing}
        />
        
        {/* Send Button */}
        <button
          type="submit"
          disabled={!input.trim() || isProcessing}
          className={`send-btn ${isProcessing ? 'loading' : ''}`}
        >
          {isProcessing ? (
            <div className="loading-spinner" />
          ) : (
            <svg viewBox="0 0 24 24">
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
            </svg>
          )}
        </button>
      </form>
    </div>
    </>
  )
}

export default InputArea
