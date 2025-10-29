import './ExamplesSection.css'

interface ExamplesSectionProps {
  onExampleClick: (text: string) => void
}

const examples = [
  {
    key: 'universal',
    title: 'Universal Reasoning',
    text: 'All cats are mammals. Fluffy is a cat. What mammals do we have?',
    description: 'Basic universal reasoning with multiple instances'
  },
  {
    key: 'family',
    title: 'Family Relations',
    text: 'Mary is parent of Alice. Mary is parent of Bob. Who are Mary\'s children?',
    description: 'Family relationship analysis'
  },
  {
    key: 'simple',
    title: 'Simple Facts',
    text: 'John is a teacher. Sarah is a student. Who are teachers?',
    description: 'Simple fact-based reasoning'
  },
  {
    key: 'basic',
    title: 'Basic Logic',
    text: 'Alice is a student. Bob is a student. Who are students?',
    description: 'Basic student identification'
  }
]

const ExamplesSection = ({ onExampleClick }: ExamplesSectionProps) => {
  return (
    <div className="examples-section">
      <div className="examples-title">💡 Try these examples:</div>
      <div className="examples-container">
        {examples.map((example) => (
          <div
            key={example.key}
            className="example-box"
            onClick={() => onExampleClick(example.text)}
          >
            <div className="example-title">{example.title}</div>
            <div className="example-text">{example.text}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default ExamplesSection
