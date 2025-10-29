import ChatInterface from './components/ChatInterface'
import Header from './components/Header'

function App() {
  const handleReset = async () => {
    if (confirm('Are you sure you want to start a new chat? This will clear all stored facts and start fresh.')) {
      try {
        await fetch('http://localhost:8002/kb/clear', { method: 'DELETE' })
        window.location.reload()
      } catch (error) {
        console.error('Error clearing knowledge base:', error)
      }
    }
  }

  return (
    <div className="app-container">
      {/* Header */}
      <Header onReset={handleReset} />

      {/* Main chat container */}
      <div className="chat-container">
        <ChatInterface />
      </div>
    </div>
  )
}

export default App
