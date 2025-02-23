import tkinter as tk
from tkinter import scrolledtext, ttk
import requests
import json
import threading
import queue
import os
import re

class DeepSeekChatGUI:
    def __init__(self, master):
        self.master  = master
        master.title("DeepSeek-R1 Chat")
        
        # API 配置
        self.api_key  = os.getenv("NIM_DS_API_KEY")  # 替换为你的实际API密钥
        self.api_endpoint  = "https://integrate.api.nvidia.com/v1/chat/completions" 
        self.stream_queue  = queue.Queue()  # 流式响应队列
        self.current_streaming  = False     # 流式会话状态标记
        
        # 界面布局
        self.setup_ui() 
        self.master.after(100,  self.update_gui_from_stream)   # 启动队列监听[5]()

    def setup_ui(self):
        # 输入框
        self.input_label  = tk.Label(self.master,  text="输入问题:")
        self.input_label.pack(pady=5) 
        self.input_entry  = tk.Entry(self.master,  width=60)
        self.input_entry.pack(pady=5) 
        
        # 发送按钮
        self.send_button  = tk.Button(self.master,  text="发送", command=self.start_stream_thread) 
        self.send_button.pack(pady=5) 
        
        # 取消按钮
        self.stop_button  = tk.Button(self.master,  text="停止", state=tk.DISABLED, command=self.stop_stream) 
        self.stop_button.pack(pady=5) 
        
        # 响应显示区域
        self.response_area  = scrolledtext.ScrolledText(self.master,  width=70, height=20)
        self.response_area.pack(pady=5) 
        
        # 进度条
        self.progress  = ttk.Progressbar(self.master,  mode='indeterminate')
        
        # 设置标签样式
        self.response_area.tag_config("bold",  font=("Arial", 10, "bold"))
        self.response_area.tag_config("italic",  font=("Arial", 10, "italic")) 
        self.response_area.tag_config("code",  background="#f0f0f0", relief="groove")

    def start_stream_thread(self):
        """启动流式请求线程"""
        if not self.input_entry.get(): 
            return
        
        self.current_streaming  = True
        self.send_button.config(state=tk.DISABLED) 
        self.stop_button.config(state=tk.NORMAL) 
        self.progress.pack(pady=5) 
        self.progress.start() 
        
        # 创建流式线程[3]()
        threading.Thread(
            target=self.stream_query, 
            args=(self.input_entry.get(),), 
            daemon=True
        ).start()

    def stream_query(self, query):
        """流式API请求核心逻辑"""
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.api_key}", 
            "Content-Type": "application/json"
        }
        payload = {
            "model":"deepseek-ai/deepseek-r1",
            "messages": [{"role": "user", "content": query}],
            "temperature": 0.7,
            "top_p": 0.7,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "max_tokens": 4096,
            "stop": ["string"],
            "stream": True  # 启用流式模式
        }
        
        try:
            with requests.post(self.api_endpoint,  headers=headers, json=payload, stream=True) as response:
                response.raise_for_status() 
                full_reply = ""
                markdown_buffer = ""
                in_code_block = False  # 代码块状态跟踪 
                last_valid_pos = 0     # 最后有效渲染位置 
                # 逐块解析流式响应
                for chunk in response.iter_lines(): 
                    if not self.current_streaming: 
                        break  # 用户中断
                    
                    if chunk:
                        decoded_chunk = chunk.decode('utf-8').lstrip('data:  ')
                        if decoded_chunk == "[DONE]":
                            break
                        
                        try:
                            chunk_data = json.loads(decoded_chunk) 
                            content = chunk_data['choices'][0]['delta'].get('content', '')
                            markdown_buffer += content
                            # 只有当buffer中满一行了才入队列
                            if "\n" in markdown_buffer:
                                segments = markdown_buffer.split("\n")
                                for seg in segments[:-1]:
                                    if seg.startswith("```") and in_code_block:
                                        in_code_block = False
                                    self.stream_queue.put(('chunk', seg+'\n'))# 入队更新
                                markdown_buffer = segments[-1]

                            # 保持返回内容的完整性
                            full_reply += content

                        except json.JSONDecodeError:
                            pass
                
                # 最终完整响应入队
                self.stream_queue.put(("full", full_reply))
                
        except Exception as e:
            self.stream_queue.put(("error",  f"API错误: {str(e)}"))
            
    

    def _parse_markdown(self, text):
        # 增量式Markdown解析 
        self.response_area.config(state=tk.NORMAL) 
        
        # 处理标题行
        bold_matches = list(re.finditer(r'\*\*(.*?)\*\*',  text))
        if not any(bold_matches):
            # 处理普通文本
            self.response_area.insert(tk.END,  text)
        else:
            # 处理加粗文本
            t_start = 0
            for match in bold_matches:
                m_start,m_end = match.span()
                self.response_area.insert(tk.END,  text[t_start:m_start])
                t_start = m_end
                start = self.response_area.index("end-1c") 
                self.response_area.insert(tk.END,  match.group(1)) 
                end = self.response_area.index("end-1c") 
                self.response_area.tag_add("bold",  start, end)
            self.response_area.insert(tk.END,  text[t_start:])
        # 处理代码块（需累积多行判断）
        if '``````' in text:
            code_start = self.response_area.index("end-1c  linestart")
            self.code_block_active  = not getattr(self, 'code_block_active', False)
            if self.code_block_active: 
                self.response_area.insert(tk.END,  "\n")
            else:
                self.response_area.tag_add("code",  code_start, "end-1c")
        

        self.response_area.config(state=tk.DISABLED) 

    def update_gui_from_stream(self):
        """实时更新流式响应到 GUI"""
        while True:
            try:
                # 从队列获取数据（非阻塞模式）
                msg_type,content = self.stream_queue.get_nowait() 
                
                if msg_type == 'chunk':
                    # 实时显示 chunk 内容
                    self._parse_markdown(content)
                elif msg_type == 'full':
                    # 最终状态处理（如解锁输入框）
                    self.input_entry.config(state=tk.NORMAL) 
                    self.send_button.config(state=tk.NORMAL) 
                    self.current_streaming  = False
                    self.stream_queue.task_done() 
                    break
                elif msg_type == 'error':
                    self.response_area.insert(tk.END,  f"\n错误: {content}\n")

            except queue.Empty:
                # 队列无数据时退出循环,100ms轮询一次
                self.master.after(100,  self.update_gui_from_stream) 
                break

    # def display_error(self, error_msg):
    #     """统一错误显示处理"""
    #     self.response_area.config(state=tk.NORMAL) 
    #     self.response_area.insert(tk.END,  f"\n【错误】{error_msg}\n")
    #     self.response_area.config(state=tk.DISABLED) 
        
    #     # 恢复界面交互状态
    #     self.input_entry.config(state=tk.NORMAL) 
    #     self.send_button.config(state=tk.NORMAL) 
    #     self.current_streaming  = False

    # def append_stream(self, content):
    #     """追加流式内容"""
    #     self.response_area.config(state=tk.NORMAL) 
    #     self.response_area.insert(tk.END,  content)
    #     self.response_area.see(tk.END)   # 自动滚动到底部
    #     self.response_area.config(state=tk.DISABLED) 
        
    def finalize_response(self, full_reply):
        """完成响应处理"""
        self.progress.stop() 
        self.progress.pack_forget() 
        self.response_area.insert(tk.END,  "\n\n")
        self.send_button.config(state=tk.NORMAL) 
        self.stop_button.config(state=tk.DISABLED) 
        self.current_streaming  = False
        self.input_entry.delete(0,  tk.END)
        
    def stop_stream(self):
        """用户中断流式响应"""
        self.current_streaming  = False
        self.stream_queue.queue.clear() 
        self.finalize_response("[ 用户中断]")
        
if __name__ == "__main__":
    root = tk.Tk()
    gui = DeepSeekChatGUI(root)
    root.mainloop()



    