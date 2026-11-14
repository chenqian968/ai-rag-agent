// 前端交互逻辑:上传文档、对话、多轮记忆
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const fileInput = document.getElementById('file-input');
const uploadStatus = document.getElementById('upload-status');
const resetBtn = document.getElementById('reset-btn');

// 多轮对话历史(发送给后端,实现记忆)
let history = [];

// 清空历史
function clearHistory() {
    history = [];
}

// 添加消息到界面
function addMessage(role, content, isTyping = false) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = role === 'user' ? '我' : 'AI';

    const bubble = document.createElement('div');
    bubble.className = 'bubble';

    if (isTyping) {
        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            indicator.appendChild(dot);
        }
        bubble.appendChild(indicator);
    } else {
        bubble.textContent = content;
    }

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(bubble);
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return { msgDiv, bubble };
}

// 发送消息
async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    // 显示用户消息
    addMessage('user', text);
    chatInput.value = '';
    chatInput.style.height = 'auto';
    sendBtn.disabled = true;

    // 显示"正在输入"指示
    const { msgDiv, bubble } = addMessage('assistant', '', true);

    // 构造请求
    const reqHistory = history.map(m => ({ role: m.role, content: m.content }));

    try {
        const resp = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text, history: reqHistory }),
        });

        const data = await resp.json();

        // 移除 typing 指示,显示真实回答
        bubble.innerHTML = '';
        bubble.textContent = data.answer || (data.error ? `出错: ${data.error}` : '(无回答)');

        // 更新历史
        history.push({ role: 'user', content: text });
        history.push({ role: 'assistant', content: data.answer || '' });
    } catch (err) {
        bubble.innerHTML = '';
        bubble.textContent = `网络错误: ${err.message}`;
    }

    sendBtn.disabled = false;
    chatMessages.scrollTop = chatMessages.scrollHeight;
    chatInput.focus();
}

// 上传文档
async function uploadFile(file) {
    if (!file) return;

    uploadStatus.textContent = `正在上传 ${file.name} ...`;
    const formData = new FormData();
    formData.append('file', file);

    try {
        const resp = await fetch('/api/ingest', {
            method: 'POST',
            body: formData,
        });
        const data = await resp.json();

        if (resp.ok) {
            uploadStatus.textContent = `✅ 已导入 ${file.name}(${data.chunks} 个文本块)`;
        } else {
            uploadStatus.textContent = `❌ ${data.detail || '导入失败'}`;
        }
    } catch (err) {
        uploadStatus.textContent = `❌ 网络错误: ${err.message}`;
    }

    fileInput.value = '';
}

// 清空知识库
async function resetKnowledge() {
    if (!confirm('确定清空知识库吗?此操作不可恢复。')) return;

    uploadStatus.textContent = '正在清空...';
    try {
        const resp = await fetch('/api/reset', { method: 'POST' });
        const data = await resp.json();
        uploadStatus.textContent = data.message || '已清空';
        clearHistory();
    } catch (err) {
        uploadStatus.textContent = `❌ 网络错误: ${err.message}`;
    }
}

// 事件绑定
sendBtn.addEventListener('click', sendMessage);

chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// 自动调整输入框高度
chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
});

fileInput.addEventListener('change', (e) => {
    uploadFile(e.target.files[0]);
});

resetBtn.addEventListener('click', resetKnowledge);
