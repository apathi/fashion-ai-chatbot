/**
 * Fashion AI Chatbot - Frontend JavaScript
 *
 * Handles UI interactions, API calls, and message rendering.
 */

// API Base URL
const API_BASE = window.location.origin;

// DOM Elements
const messagesContainer = document.getElementById('messagesContainer');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');
const ragToggle = document.getElementById('ragToggle');
const infoBanner = document.getElementById('infoBanner');
const modeTitle = document.getElementById('modeTitle');
const modeContext = document.getElementById('modeContext');
const modeDescription = document.getElementById('modeDescription');
const loadingIndicator = document.getElementById('loadingIndicator');
const questionsGrid = document.getElementById('questionsGrid');
const compareButton = document.getElementById('compareButton');
const clearButton = document.getElementById('clearButton');

// State
let isLoading = false;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadSampleQuestions();
    setupEventListeners();
    updateModeUI();
});

// Event Listeners
function setupEventListeners() {
    sendButton.addEventListener('click', handleSend);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });

    ragToggle.addEventListener('change', updateModeUI);
    compareButton.addEventListener('click', handleCompare);
    clearButton.addEventListener('click', handleClear);

    // Auto-resize textarea
    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        userInput.style.height = userInput.scrollHeight + 'px';
    });
}

// Update UI based on mode
function updateModeUI() {
    const isRAG = ragToggle.checked;

    if (isRAG) {
        infoBanner.style.background = '#e8f4fd';
        infoBanner.style.borderLeftColor = '#2196f3';
        modeTitle.textContent = 'RAG Mode Active:';
        modeContext.textContent = 'Responses are grounded in 82 curated 2023 fashion trends.';
        modeDescription.textContent = 'Each answer includes sources and citations.';
    } else {
        infoBanner.style.background = '#fff3e0';
        infoBanner.style.borderLeftColor = '#ff9800';
        modeTitle.textContent = 'Basic Mode Active:';
        modeContext.textContent = 'Using general knowledge without specific trend data.';
        modeDescription.textContent = 'Responses may be generic without specific trend references.';
    }
}

// Load sample questions
async function loadSampleQuestions() {
    try {
        const response = await fetch(`${API_BASE}/api/sample-questions`);
        const data = await response.json();

        questionsGrid.innerHTML = '';
        data.questions.forEach(question => {
            const chip = document.createElement('div');
            chip.className = 'question-chip';
            chip.textContent = question;
            chip.addEventListener('click', () => {
                userInput.value = question;
                userInput.focus();
            });
            questionsGrid.appendChild(chip);
        });
    } catch (error) {
        console.error('Failed to load sample questions:', error);
    }
}

// Handle send message
async function handleSend() {
    const message = userInput.value.trim();
    if (!message || isLoading) return;

    const mode = ragToggle.checked ? 'rag' : 'basic';

    // Clear input
    userInput.value = '';
    userInput.style.height = 'auto';

    // Add user message
    addMessage(message, 'user');

    // Show loading
    setLoading(true);

    try {
        const response = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message, mode, top_k: 3 }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Add bot response
        addMessage(data.response, 'bot', data.sources, data.mode);

    } catch (error) {
        console.error('Error:', error);
        addMessage(
            'Sorry, I encountered an error. Please try again.',
            'bot',
            null,
            'error'
        );
    } finally {
        setLoading(false);
    }
}

// Handle compare mode
async function handleCompare() {
    const message = userInput.value.trim();
    if (!message || isLoading) return;

    // Clear input
    userInput.value = '';
    userInput.style.height = 'auto';

    // Add user message
    addMessage(message, 'user');

    // Show loading
    setLoading(true);

    try {
        const response = await fetch(`${API_BASE}/api/compare`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message, top_k: 3 }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Add comparison view
        addComparisonMessage(data);

    } catch (error) {
        console.error('Error:', error);
        addMessage(
            'Sorry, I encountered an error. Please try again.',
            'bot',
            null,
            'error'
        );
    } finally {
        setLoading(false);
    }
}

// Handle clear chat
async function handleClear() {
    if (!confirm('Clear chat history?')) return;

    try {
        await fetch(`${API_BASE}/api/history/clear`, {
            method: 'POST',
        });

        // Clear messages except welcome
        const messages = messagesContainer.querySelectorAll('.message, .comparison-container');
        messages.forEach(msg => msg.remove());

    } catch (error) {
        console.error('Error clearing history:', error);
    }
}

// Add message to chat
function addMessage(content, sender, sources = null, mode = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;

    messageDiv.appendChild(contentDiv);

    // Add footer with mode indicator
    if (sender === 'bot' && mode) {
        const footer = document.createElement('div');
        footer.className = 'message-footer';
        footer.textContent = mode === 'rag' ? '🔍 RAG Enhanced' : '💬 Basic Response';
        messageDiv.appendChild(footer);
    }

    // Add sources if available
    if (sources && sources.length > 0) {
        const sourcesDiv = createSourcesElement(sources);
        messageDiv.appendChild(sourcesDiv);
    }

    messagesContainer.appendChild(messageDiv);
    scrollToBottom();
}

// Create sources element
function createSourcesElement(sources) {
    const sourcesDiv = document.createElement('div');
    sourcesDiv.className = 'sources';

    const title = document.createElement('div');
    title.className = 'sources-title';
    title.textContent = '📚 Sources Used';
    sourcesDiv.appendChild(title);

    sources.forEach(source => {
        const sourceItem = document.createElement('div');
        sourceItem.className = 'source-item';

        const name = document.createElement('div');
        name.className = 'source-name';
        name.textContent = source.trend_name || 'Fashion Trend';
        sourceItem.appendChild(name);

        if (source.URL) {
            const link = document.createElement('a');
            link.className = 'source-link';
            link.href = source.URL;
            link.target = '_blank';
            link.textContent = '🔗 View Source';
            sourceItem.appendChild(link);
        }

        if (source.similarity !== undefined) {
            const similarity = document.createElement('div');
            similarity.className = 'source-similarity';
            similarity.textContent = `Relevance: ${(source.similarity * 100).toFixed(1)}%`;
            sourceItem.appendChild(similarity);
        }

        sourcesDiv.appendChild(sourceItem);
    });

    return sourcesDiv;
}

// Add comparison message
function addComparisonMessage(data) {
    const comparisonDiv = document.createElement('div');
    comparisonDiv.className = 'comparison-container';

    const title = document.createElement('div');
    title.className = 'comparison-title';
    title.textContent = '⚖️ RAG vs Basic Comparison';
    comparisonDiv.appendChild(title);

    const grid = document.createElement('div');
    grid.className = 'comparison-grid';

    // Basic response
    const basicSide = document.createElement('div');
    basicSide.className = 'comparison-side basic';
    basicSide.innerHTML = `
        <div class="comparison-label">❌ Basic (No Context)</div>
        <div class="comparison-text">${data.basic_response}</div>
    `;
    grid.appendChild(basicSide);

    // RAG response
    const ragSide = document.createElement('div');
    ragSide.className = 'comparison-side rag';
    ragSide.innerHTML = `
        <div class="comparison-label">✅ RAG Enhanced</div>
        <div class="comparison-text">${data.rag_response}</div>
    `;
    grid.appendChild(ragSide);

    comparisonDiv.appendChild(grid);

    // Add sources
    if (data.sources && data.sources.length > 0) {
        const sourcesDiv = createSourcesElement(data.sources);
        comparisonDiv.appendChild(sourcesDiv);
    }

    messagesContainer.appendChild(comparisonDiv);
    scrollToBottom();
}

// Set loading state
function setLoading(loading) {
    isLoading = loading;
    loadingIndicator.style.display = loading ? 'flex' : 'none';
    sendButton.disabled = loading;
    compareButton.disabled = loading;

    if (loading) {
        scrollToBottom();
    }
}

// Scroll to bottom
function scrollToBottom() {
    setTimeout(() => {
        messagesContainer.parentElement.scrollTop = messagesContainer.parentElement.scrollHeight;
    }, 100);
}

// Utility: Format timestamp
function formatTime() {
    const now = new Date();
    return now.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit'
    });
}
