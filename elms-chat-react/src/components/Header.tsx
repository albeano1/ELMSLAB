import './Header.css'

interface HeaderProps {
  onReset?: () => void
}

const Header = ({ onReset }: HeaderProps) => {
  return (
    <div className="header">
      <div className="header-left">
        <button className="reset-btn glass-button" onClick={onReset} title="New Chat">
          <svg viewBox="0 0 24 24">
            <path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/>
          </svg>
        </button>
        <h1>ELMS Chat</h1>
      </div>
    </div>
  )
}

export default Header
