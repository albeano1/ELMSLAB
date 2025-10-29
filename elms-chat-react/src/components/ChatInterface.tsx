import { useState, useRef, useEffect } from 'react'
import MessageList from './MessageList'
import InputArea from './InputArea'
import ExamplesSection from './ExamplesSection'
import './ChatInterface.css'

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

const API_BASE = 'http://localhost:8002'

const ChatInterface = () => {
  const [messages, setMessages] = useState<Message[]>([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [showExamples, setShowExamples] = useState(true)
  const [exampleText, setExampleText] = useState<string>('')
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Load messages from localStorage on component mount
  useEffect(() => {
    const savedMessages = localStorage.getItem('elms-chat-messages')
    if (savedMessages) {
      try {
        const parsedMessages = JSON.parse(savedMessages)
        setMessages(parsedMessages)
        setShowExamples(parsedMessages.length === 0)
      } catch (error) {
        console.error('Error loading saved messages:', error)
      }
    }
  }, [])

  // Save messages to localStorage whenever messages change
  useEffect(() => {
    localStorage.setItem('elms-chat-messages', JSON.stringify(messages))
  }, [messages])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleExampleClick = (text: string) => {
    setExampleText(text)
  }

  const handleSendMessage = async (content: string) => {
    console.log('🚀 handleSendMessage called with content:', content)
    if (!content.trim() || isProcessing) {
      console.log('❌ handleSendMessage rejected - empty content or processing')
      return
    }

    console.log('✅ handleSendMessage processing...')
    setIsProcessing(true)
    setShowExamples(false)
    setExampleText('') // Clear example text when sending

    // Store uploaded file for processing before clearing
    const fileToProcess = uploadedFile
    
    // Add user message with image if uploaded
    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: content.trim(),
      timestamp: new Date(),
      imageUrl: fileToProcess && fileToProcess.type.startsWith('image/') 
        ? URL.createObjectURL(fileToProcess) 
        : undefined
    }
    setMessages(prev => [...prev, userMessage])
    
    // Clear the uploaded file immediately so preview disappears from input bar
    setUploadedFile(null)

    try {
      let response
      
      // Check if there's an uploaded file to process
      if (fileToProcess) {
        // First, upload the file and extract text
        const uploadFormData = new FormData()
        uploadFormData.append('file', fileToProcess)
        
        const uploadResponse = await fetch(`${API_BASE}/upload?t=${Date.now()}`, {
          method: 'POST',
          body: uploadFormData
        })
        
        const uploadResult = await uploadResponse.json()
        console.log('📤 Upload result:', uploadResult)
        
        // Always use regular pipeline for rich formatting
        // Add extracted facts to knowledge base if any
        if (uploadResult.success && uploadResult.facts && uploadResult.facts.length > 0) {
          console.log(`✅ Adding ${uploadResult.facts.length} facts to knowledge base`)
          for (const fact of uploadResult.facts) {
            await fetch(`${API_BASE}/kb/add_fact`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                text: fact,
                confidence: 0.95
              })
            })
          }
        }
        
        // Send through regular inference endpoint for rich formatting
        console.log('📤 Sending visual query to /infer:', {
          premises: uploadResult.facts || [],
          conclusion: content.trim()
        })
        response = await fetch(`${API_BASE}/infer?t=${Date.now()}`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            premises: uploadResult.facts || [],
            conclusion: content.trim()
          })
        })
      } else {
        // Parse the message to extract premises and conclusion
        const { premises, conclusion } = parseMessage(content.trim())
        console.log('Parsed message:', { premises, conclusion })
        
        // Store premises in knowledge base
        if (premises.length > 0) {
          for (const premise of premises) {
            await fetch(`${API_BASE}/kb/add_fact`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                text: premise,
                confidence: 0.95
              })
            })
          }
        }
        
        // Send to API
        response = await fetch(`${API_BASE}/infer?t=${Date.now()}`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            premises: premises,
            conclusion: conclusion
          })
        })
      }

      // Check for errors before parsing JSON
      if (!response.ok) {
        let errorMessage = `HTTP error! status: ${response.status}`
        try {
          const errorData = await response.json()
          if (errorData.detail) {
            errorMessage = errorData.detail
          }
        } catch (e) {
          // If we can't parse JSON, use the default error
        }
        throw new Error(errorMessage)
      }
      
      const result = await response.json()
      console.log('📥 API response:', result)
      console.log('result.valid:', result.valid)
      console.log('result.answer:', result.answer)
      console.log('result.reasoning_steps:', result.reasoning_steps)
      console.log('result.formal_steps:', result.formal_steps)
      console.log('result.parsed_premises:', result.parsed_premises)
      
      // Handle visual reasoning response format
      const isValid = result.valid !== undefined ? result.valid : result.success
      const answer = result.answer || (isValid ? 'Yes' : null)
      
      // Format answer for visual reasoning (array of strings)
      let formattedAnswer = answer
      if (Array.isArray(answer) && answer.length > 0) {
        formattedAnswer = answer.join(', ')
      } else if (Array.isArray(answer)) {
        formattedAnswer = 'No results found'
      }
      
      // Add assistant response with all API data
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: isValid && formattedAnswer ? formattedAnswer : 'No valid conclusion found',
        timestamp: new Date(),
        reasoningSteps: result.reasoning_steps,
        formalSteps: result.formal_steps,
        parsedPremises: result.parsed_premises || result.premises_used,
        parsedConclusion: result.parsed_conclusion || result.question,
        queryTime: result.query_time || result.reasoning_steps,
        confidence: result.confidence || (isValid ? 0.95 : 0.0),
        valid: isValid,
        answer: formattedAnswer,
        explanation: result.explanation,
        // Add all other fields from API response
        logic_type: result.logic_type,
        confidence_level: result.confidence_level,
        vectionary_enhanced: result.vectionary_enhanced,
        conclusions_count: result.conclusions_count
      }
      
      setMessages(prev => [...prev, assistantMessage])
      
    } catch (error) {
      console.error('Error:', error)
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: `Sorry, I encountered an error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsProcessing(false)
    }
  }

  const handleFileUpload = async (file: File) => {
    if (!file) return
    
    console.log('📁 File uploaded:', file.name)
    
    // Store the file for later processing - preview will show in InputArea
    setUploadedFile(file)
  }

  const handleReset = async () => {
    console.log('handleReset called')
    setMessages([])
    setShowExamples(true)
    setUploadedFile(null)
    localStorage.removeItem('elms-chat-messages')
    
    // Clear knowledge base
    try {
      console.log('Clearing knowledge base with DELETE method...')
      const response = await fetch(`${API_BASE}/kb/clear?t=${Date.now()}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        }
      })
      console.log('Response status:', response.status)
      const result = await response.json()
      console.log('Knowledge base clear result:', result)
    } catch (error) {
      console.error('Error clearing knowledge base:', error)
    }
  }

  const parseMessage = (message: string) => {
    const questionWords = ['what', 'who', 'where', 'when', 'why', 'how', 'does', 'is', 'are', 'can', 'will']
    const sentences = message.split(/[.!?]+/).filter(s => s.trim())
    
    let conclusion = ''
    let premises: string[] = []
    
    for (let sentence of sentences) {
      const trimmed = sentence.trim()
      if (!trimmed) continue
      
      const isQuestion = questionWords.some(word => 
        trimmed.toLowerCase().startsWith(word)
      ) || trimmed.endsWith('?')
      
      if (isQuestion) {
        conclusion = trimmed + (trimmed.endsWith('?') ? '' : '?')
      } else {
        premises.push(trimmed + (trimmed.endsWith('.') ? '' : '.'))
      }
    }
    
    // Only use the last sentence as conclusion if it's clearly a question
    if (!conclusion && premises.length > 0) {
      const lastSentence = premises[premises.length - 1]
      const isLastQuestion = questionWords.some(word => 
        lastSentence.toLowerCase().startsWith(word)
      ) || lastSentence.endsWith('?')
      
      if (isLastQuestion) {
        conclusion = premises.pop() || ''
      }
    }
    
    return { premises, conclusion }
  }

  return (
    <>
      {/* Messages Area */}
      <div className="chat-messages">
        <MessageList messages={messages} isProcessing={isProcessing} />
          {showExamples && messages.length === 0 && (
            <ExamplesSection onExampleClick={handleExampleClick} />
          )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <InputArea
        onSendMessage={handleSendMessage}
        isProcessing={isProcessing}
        onReset={handleReset}
        exampleText={exampleText}
        onFileUpload={handleFileUpload}
        uploadedFile={uploadedFile}
      />
    </>
  )
}

export default ChatInterface
