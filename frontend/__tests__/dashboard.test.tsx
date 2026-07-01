import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';

// 1. Mock Monaco Editor
jest.mock('@monaco-editor/react', () => {
  return function MockEditor({ value, onChange }: any) {
    return (
      <textarea
        data-testid="monaco-editor-textarea"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    );
  };
});

// 2. Mock React Flow
jest.mock('@xyflow/react', () => {
  return {
    ReactFlow: ({ children }: any) => <div data-testid="react-flow-graph">{children}</div>,
    Background: () => <div />,
    Controls: () => <div />,
  };
});

// 3. Mock Recharts
jest.mock('recharts', () => {
  const OriginalModule = jest.requireActual('recharts');
  return {
    ...OriginalModule,
    ResponsiveContainer: ({ children }: any) => <div>{children}</div>,
  };
});

// 4. Test Mock Components
import CitationDrawer from '../src/components/playground/CitationDrawer';
import DiffViewer from '../src/components/playground/DiffViewer';
import RetrievalInspector from '../src/components/playground/RetrievalInspector';
import PipelineEditorModal from '../src/components/pipelines/PipelineEditorModal';

describe('NeuroFlow Frontend Suite', () => {
  
  test('citation drawer renders source data and triggers close', () => {
    const mockOnClose = jest.fn();
    const mockSource = {
      document_name: 'test_doc.pdf',
      page: 4,
      chunk_id: 'chk-123',
      text: 'Attention mechanisms allow modeling dependencies.',
      metadata: { score: 0.95 }
    };

    const { rerender } = render(
      <CitationDrawer isOpen={true} onClose={mockOnClose} source={mockSource} />
    );

    expect(screen.getByText('test_doc.pdf')).toBeInTheDocument();
    expect(screen.getByText('Attention mechanisms allow modeling dependencies.')).toBeInTheDocument();
    expect(screen.getByText('chk-123')).toBeInTheDocument();

    const closeBtn = screen.getByRole('button');
    fireEvent.click(closeBtn);
    expect(mockOnClose).toHaveBeenCalled();
  });

  test('diff viewer highlights insertions and deletions', () => {
    const answerA = 'Transformers use self-attention mechanism';
    const answerB = 'Transformers use recursive self-attention mechanism';

    render(<DiffViewer answerA={answerA} answerB={answerB} />);

    // Check if the word "recursive" is highlighted as added
    const addedWord = screen.getByText('recursive');
    expect(addedWord).toBeInTheDocument();
    expect(addedWord).toHaveClass('bg-emerald-100');
  });

  test('retrieval inspector renders xyflow react flow graph', () => {
    render(<RetrievalInspector chunkCount={5} />);
    expect(screen.getByTestId('react-flow-graph')).toBeInTheDocument();
  });

  test('pipeline config editor validates JSON schema inline', async () => {
    const mockOnSave = jest.fn();
    const mockOnClose = jest.fn();

    render(
      <PipelineEditorModal
        isOpen={true}
        onClose={mockOnClose}
        onSave={mockOnSave}
      />
    );

    const textarea = screen.getByTestId('monaco-editor-textarea');
    expect(textarea).toBeInTheDocument();

    // Trigger validation error on invalid JSON structure
    fireEvent.change(textarea, { target: { value: '{ invalid_json }' } });
    expect(screen.getByText(/Expected property name/i)).toBeInTheDocument();

    // Trigger schema error on missing generation block
    const missingGeneration = JSON.stringify({
      name: 'broken-v1',
      description: 'broken description',
      ingestion: {},
      retrieval: {},
      evaluation: {}
    });
    fireEvent.change(textarea, { target: { value: missingGeneration } });
    expect(screen.getByText(/Schema error: Missing root attribute "generation"/i)).toBeInTheDocument();
  });
});
