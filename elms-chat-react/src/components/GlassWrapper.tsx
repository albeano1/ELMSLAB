import React from 'react'
import './GlassWrapper.css'

interface GlassWrapperProps {
  children: React.ReactNode
  className?: string
  variant?: 'default' | 'message' | 'input' | 'header'
  intensity?: 'light' | 'medium' | 'strong'
}

const GlassWrapper: React.FC<GlassWrapperProps> = ({ 
  children, 
  className = '', 
  variant = 'default',
  intensity = 'medium'
}) => {
  const getVariantClasses = () => {
    switch (variant) {
      case 'message':
        return 'glass-message'
      case 'input':
        return 'glass-input'
      case 'header':
        return 'glass-header'
      default:
        return 'glass-default'
    }
  }

  const getIntensityClasses = () => {
    switch (intensity) {
      case 'light':
        return 'glass-light'
      case 'strong':
        return 'glass-strong'
      default:
        return 'glass-medium'
    }
  }

  return (
    <div 
      className={`glass-wrapper ${getVariantClasses()} ${getIntensityClasses()} ${className}`}
    >
      {children}
    </div>
  )
}

export default GlassWrapper
