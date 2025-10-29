import { useState } from 'react'
import './MessageList.css'

interface Message {
  id: string
  type: 'user' | 'assistant'
  content: string
  timestamp: Date
  imageUrl?: string
  reasoningSteps?: string[]
  formalSteps?: string[]
  parsedPremises?: string[]
  parsedConclusion?: string
  queryTime?: number
  confidence?: number
  valid?: boolean
  answer?: string
  explanation?: string
  logic_type?: string
  confidence_level?: string
  vectionary_enhanced?: boolean
  conclusions_count?: number
}

interface MessageListProps {
  messages: Message[]
  isProcessing?: boolean
}

const MessageList = ({ messages, isProcessing = false }: MessageListProps) => {
  const [expandedMessages, setExpandedMessages] = useState<Set<string>>(new Set())

  const toggleExpanded = (messageId: string) => {
    setExpandedMessages(prev => {
      const newSet = new Set(prev)
      if (newSet.has(messageId)) {
        newSet.delete(messageId)
      } else {
        newSet.add(messageId)
      }
      return newSet
    })
  }

  const formatContent = (message: Message) => {
    let content = ''
    
    if (message.type === 'assistant') {
      // Only show special formatting if this is a result message (has valid property set)
      if (message.valid !== undefined) {
        if (message.valid) {
          if (message.answer) {
            content = `
              <strong>✅ Answer:</strong> ${message.answer}
              <br><br>
              <strong>📊 Found:</strong> ${message.conclusions_count || 0} result(s)
            `
          } else {
            content = `
              <strong>✅ Conclusion:</strong> Yes
              <br><br>
              <strong>🎯 Confidence:</strong> ${((message.confidence || 0) * 100).toFixed(1)}%
            `
          }
        } else {
          content = `
            <strong>❌ Conclusion:</strong> No valid conclusion found
            <br><br>
            <strong>🎯 Confidence:</strong> ${((message.confidence || 0) * 100).toFixed(1)}%
          `
        }
      } else {
        // This is just a regular message (like upload confirmation), display content as-is
        content = message.content
      }
    } else {
      content = message.content
    }
    
    return content
  }

  return (
    <>
      {messages.map((message) => (
        <div key={message.id} data-message-id={message.id} className={`message-wrapper ${message.type === 'user' ? 'user-message' : 'assistant-message'}`}>
        <div className="message-glass">
          <div className="message-header">
            {message.type === 'user' ? 'ELMS User' : 'ELMS ENHANCED LOGIC MODELING SYSTEM'}
          </div>
          {message.imageUrl && (
            <div className="message-image-preview">
              <img src={message.imageUrl} alt="Uploaded" style={{ maxWidth: '300px', maxHeight: '200px', borderRadius: '8px', marginBottom: '10px' }} />
            </div>
          )}
          <div 
            className="message-content"
            dangerouslySetInnerHTML={{ __html: formatContent(message) }}
          />
          
          {/* Reasoning Steps */}
          {message.reasoningSteps && message.reasoningSteps.length > 0 && (
            <div>
              <button
                onClick={() => toggleExpanded(message.id)}
                className="reasoning-toggle-btn"
              >
                {expandedMessages.has(message.id) ? '▼' : '▶'} How I reached this conclusion
              </button>
              {expandedMessages.has(message.id) && (
                <div className="reasoning-steps">
                  <h4>🔍 How I reached this conclusion:</h4>
                  <ul>
                    {message.reasoningSteps.map((step, index) => (
                      <li key={index}>{step}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
          
          {/* Formal Steps */}
          {message.formalSteps && message.formalSteps.length > 0 && (
            <div>
              <button
                onClick={() => toggleExpanded(`${message.id}-formal`)}
                className="reasoning-toggle-btn"
              >
                {expandedMessages.has(`${message.id}-formal`) ? '▼' : '▶'} Formal reasoning steps
              </button>
              {expandedMessages.has(`${message.id}-formal`) && (
                <div className="reasoning-steps">
                  <h4>🔬 Formal reasoning steps:</h4>
                  <ul>
                    {message.formalSteps.map((step, index) => (
                      <li key={index}>{step}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
          
          {/* Parsed Premises */}
          {message.parsedPremises && message.parsedPremises.length > 0 && (
            <div>
              <button
                onClick={() => toggleExpanded(`${message.id}-premises`)}
                className="reasoning-toggle-btn"
              >
                {expandedMessages.has(`${message.id}-premises`) ? '▼' : '▶'} Prolog facts & rules
              </button>
              {expandedMessages.has(`${message.id}-premises`) && (
                <div className="reasoning-steps">
                  <h4>🔬 Prolog facts & rules:</h4>
                  <ul>
                    {message.parsedPremises.map((fact, index) => (
                      <li key={index}>{index + 1}. {fact}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
          
          {/* Parsed Conclusion */}
          {message.parsedConclusion && (
            <div>
              <button
                onClick={() => toggleExpanded(`${message.id}-conclusion`)}
                className="reasoning-toggle-btn"
              >
                {expandedMessages.has(`${message.id}-conclusion`) ? '▼' : '▶'} Query
              </button>
              {expandedMessages.has(`${message.id}-conclusion`) && (
                <div className="reasoning-steps">
                  <h4>❓ Query:</h4>
                  <ul>
                    <li>{message.parsedConclusion}</li>
                  </ul>
                </div>
              )}
            </div>
          )}
          
          {/* Performance Metrics */}
          {message.queryTime && (
            <div>
              <button
                onClick={() => toggleExpanded(`${message.id}-performance`)}
                className="reasoning-toggle-btn"
              >
                {expandedMessages.has(`${message.id}-performance`) ? '▼' : '▶'} Performance
              </button>
              {expandedMessages.has(`${message.id}-performance`) && (
                <div className="reasoning-steps">
                  <h4>⏱️ Performance:</h4>
                  <ul>
                    <li>Query time: {message.queryTime.toFixed(2)}s</li>
                    <li>Logic type: {message.logic_type || 'Unknown'}</li>
                    <li>Confidence level: {message.confidence_level || 'Unknown'}</li>
                    <li>Vectionary enhanced: {message.vectionary_enhanced ? 'Yes' : 'No'}</li>
                  </ul>
                </div>
              )}
            </div>
          )}
          
          {/* Detailed Explanation */}
          {message.explanation && (
            <div>
              <button
                onClick={() => toggleExpanded(`${message.id}-explanation`)}
                className="reasoning-toggle-btn"
              >
                {expandedMessages.has(`${message.id}-explanation`) ? '▼' : '▶'} Detailed explanation
              </button>
              {expandedMessages.has(`${message.id}-explanation`) && (
                <div className="reasoning-steps">
                  <h4>📝 Detailed explanation:</h4>
                  <div className="explanation-content">
                    {message.explanation}
                  </div>
                </div>
              )}
            </div>
          )}
          
          {/* Copy Button for debugging - only for assistant messages */}
          {message.type === 'assistant' && (
            <div className="copy-button-wrapper">
              <button
                onClick={(event) => {
                    console.log('Copy button clicked!')
                    
                    // Get the button element immediately before async operations
                    const button = event.currentTarget as HTMLElement
                    if (!button) {
                      console.error('Button element not found')
                      return
                    }
                    
                    const fullContent = JSON.stringify({
                      id: message.id,
                      type: message.type,
                      content: message.content,
                      valid: message.valid,
                      answer: message.answer,
                      count: message.conclusions_count,
                      confidence: message.confidence,
                      reasoningSteps: message.reasoningSteps,
                      formalSteps: message.formalSteps,
                      parsedPremises: message.parsedPremises,
                      parsedConclusion: message.parsedConclusion,
                      explanation: message.explanation,
                      queryTime: message.queryTime
                    }, null, 2)
                    
                    navigator.clipboard.writeText(fullContent).then(() => {
                      // Animate button growth with text
                      console.log('Copy successful, animating button')
                      
                      // Add success class and show text
                      button.classList.add('copy-success')
                      
                      // Create text element
                      const textElement = document.createElement('span')
                      textElement.className = 'copy-text'
                      textElement.textContent = 'Copied!'
                      
                      // Add text to wrapper
                      const wrapper = button.parentElement as HTMLElement
                      if (wrapper) {
                        wrapper.appendChild(textElement)
                      }
                      
                      console.log('Button classes:', button.className)
                      
                      // Hold the size for 0.8 seconds
                      setTimeout(() => {
                        button.classList.remove('copy-success')
                        if (textElement && textElement.parentNode) {
                          textElement.parentNode.removeChild(textElement)
                        }
                        console.log('Animation complete')
                      }, 800)
                    }).catch(() => {
                      // Animate button growth for error
                      console.log('Copy failed, animating button')
                      
                      button.classList.add('copy-error')
                      
                      // Create text element
                      const textElement = document.createElement('span')
                      textElement.className = 'copy-text'
                      textElement.textContent = 'Error!'
                      
                      // Add text to wrapper
                      const wrapper = button.parentElement as HTMLElement
                      if (wrapper) {
                        wrapper.appendChild(textElement)
                      }
                      
                      // Hold the size for 0.8 seconds
                      setTimeout(() => {
                        button.classList.remove('copy-error')
                        if (textElement && textElement.parentNode) {
                          textElement.parentNode.removeChild(textElement)
                        }
                      }, 800)
                    })
                  }}
                  className="copy-message-btn-icon"
                  title="Copy all message details for debugging"
                >
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                  </svg>
                </button>
            </div>
          )}
        </div>
        </div>
      ))}
      
      {/* Skeleton Loading */}
      {isProcessing && (
        <div className="message-wrapper assistant-message">
          <div className="message-glass">
            <div className="message-header">
              ELMS ENHANCED LOGIC MODELING SYSTEM
            </div>
            <div className="skeleton-loading">
              {/* ✅ Answer: alice */}
              <div className="skeleton-line skeleton-label-line"></div>
              <div className="skeleton-line skeleton-answer-line"></div>
              <div className="skeleton-spacer"></div>
              {/* 📊 Found: 1 result(s) */}
              <div className="skeleton-line skeleton-label-line"></div>
              <div className="skeleton-line skeleton-found-line"></div>
              {/* ▶ How I reached this conclusion */}
              <div className="skeleton-line skeleton-toggle-line"></div>
              {/* ▶ Formal reasoning steps */}
              <div className="skeleton-line skeleton-toggle-line"></div>
              {/* ▶ Prolog facts & rules */}
              <div className="skeleton-line skeleton-toggle-line"></div>
              {/* ▶ Query */}
              <div className="skeleton-line skeleton-toggle-short-line"></div>
              {/* ▶ Performance */}
              <div className="skeleton-line skeleton-toggle-short-line"></div>
              {/* ▶ Detailed explanation */}
              <div className="skeleton-line skeleton-toggle-line"></div>
            </div>
          </div>
        </div>
      )}
    </>
  )
}

export default MessageList
