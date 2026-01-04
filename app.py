"""
Metro-Power-Agent V1 - 台北捷運供電廠智慧電力維護代理系統
系統屬性：台北捷運系統處供電廠維修輔助工具
版本：v1.1 (Sidebar API Key Update)
新增功能：
    - Matplotlib 工程圖表自動生成
    - PRPD (局部放電相位圖譜) 視覺化
    - HSCB (直流斷路器) di/dt 特性曲線視覺化
    - 支援由 Sidebar 輸入 OpenAI API Key
"""

import re
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import Dict, List, Optional
import platform

import streamlit as st
# from dotenv import load_dotenv # 已移除 .env 依賴
from openai import OpenAI

# ==================== 初始化配置 ====================
# load_dotenv() # 已移除
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) # 已移除全域初始化

# 設定 Matplotlib 中文字體 - 動態檢測可用字體
def setup_chinese_font():
    """動態檢測並設定中文字體"""
    # Windows 常見中文字體
    chinese_fonts = [
        'Microsoft JhengHei',  # 微軟正黑體
        'Microsoft YaHei',     # 微軟雅黑
        'SimHei',              # 黑體
        'DFKai-SB',            # 標楷體
        'MingLiU',             # 細明體
        'PMingLiU',            # 新細明體
        'Noto Sans CJK TC',    # Google Noto
        'Arial Unicode MS'
    ]
    
    # 獲取系統所有可用字體
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 找到第一個可用的中文字體
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            return font
    
    # 如果都找不到，嘗試使用 sans-serif 並禁用 unicode minus
    plt.rcParams['axes.unicode_minus'] = False
    return None

# 設定中文字體
setup_chinese_font()

# 設定 Matplotlib 風格以適應 Streamlit 深色/淺色模式
plt.style.use('dark_background') if st.get_option("theme.base") == "dark" else plt.style.use('default')

# ==================== 常數定義 ====================
MAX_REASONING_STEPS = 12
MODEL_NAME = "gpt-4.0-mini" # 或使用 gpt-3.5-turbo / gpt-4-turbo
MODEL_TEMPERATURE = 0.1

# ==================== 圖表生成工具函數 ====================

def generate_prpd_plot(qmax_pc: float, pattern_type: str = "void"):
    """生成局部放電 PRPD 相位圖譜 - 使用隨機數據模擬真實情況"""
    # 每次繪圖前確保字體設定正確
    setup_chinese_font()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 模擬 AC 電壓波形作為背景參考（隨機振幅變化）
    x_ac = np.linspace(0, 360, 1000)
    voltage_amplitude = qmax_pc * np.random.uniform(0.08, 0.12)  # 隨機振幅
    y_ac = np.sin(np.deg2rad(x_ac)) * voltage_amplitude
    ax.plot(x_ac, y_ac, color='gray', alpha=0.3, linestyle='--', label='AC Voltage Ref', linewidth=1.5)

    # 隨機決定事件數量（更真實的變化）
    num_points = np.random.randint(250, 450)
    
    # 根據不同缺陷類型生成不同的相位分佈
    if pattern_type == "void": # 內部空隙：發生在電壓上升緣 (0-90, 180-270)
        # 主要群聚點（隨機偏移）
        center1 = np.random.uniform(35, 55)  # 正半週
        center2 = np.random.uniform(215, 235)  # 負半週
        spread1 = np.random.uniform(12, 18)  # 隨機擴散程度
        spread2 = np.random.uniform(12, 18)
        
        # 生成主要放電事件
        phase1 = np.random.normal(center1, spread1, num_points // 2)
        phase2 = np.random.normal(center2, spread2, num_points // 2)
        phases = np.concatenate([phase1, phase2])
        
        # 模擬能量分佈（較高能量在群聚中心）
        magnitudes = []
        for phase in phases:
            # 計算距離最近群聚中心的距離
            dist = min(abs(phase - center1), abs(phase - center2), abs(phase - (center1+360)), abs(phase - (center2-360)))
            # 越接近中心，能量越高
            base_mag = qmax_pc * (0.5 + 0.5 * np.exp(-dist/10))
            # 加入隨機變化
            mag = base_mag * np.random.uniform(0.7, 1.0)
            magnitudes.append(mag)
        magnitudes = np.array(magnitudes)
        
        # 加入少量背景雜訊（5-10%）
        noise_count = int(num_points * np.random.uniform(0.05, 0.1))
        noise_phases = np.random.uniform(0, 360, noise_count)
        noise_mags = np.random.uniform(qmax_pc * 0.05, qmax_pc * 0.2, noise_count)
        phases = np.concatenate([phases, noise_phases])
        magnitudes = np.concatenate([magnitudes, noise_mags])
        
        title_suffix = f"(典型內部空隙放電 - 檢測到 {len(phases)} 個事件)"
        severity = "嚴重" if qmax_pc > 800 else "中等" if qmax_pc > 500 else "輕微"
        
    elif pattern_type == "surface": # 表面污損：通常在峰值附近
        # 在正負峰值附近形成寬廣分佈
        peak1_center = np.random.uniform(85, 95)  # 正峰值
        peak2_center = np.random.uniform(265, 275)  # 負峰值
        spread = np.random.uniform(20, 30)  # 較大擴散
        
        phase1 = np.random.normal(peak1_center, spread, num_points // 2)
        phase2 = np.random.normal(peak2_center, spread, num_points // 2)
        phases = np.concatenate([phase1, phase2])
        
        # 表面放電通常能量較分散
        magnitudes = np.random.gamma(2, qmax_pc * 0.25, num_points)  # 使用 Gamma 分佈
        magnitudes = np.clip(magnitudes, qmax_pc * 0.05, qmax_pc * 0.85)
        
        title_suffix = f"(疑似表面污損/沿面放電 - 檢測到 {len(phases)} 個事件)"
        severity = "需注意" if qmax_pc > 400 else "監測中"
        
    else: # 雜訊或未知模式
        # 完全隨機分佈，但加入一些結構
        phases = np.random.uniform(0, 360, num_points)
        
        # 模擬多個小群聚（雜訊特徵）
        num_clusters = np.random.randint(3, 8)
        for _ in range(num_clusters):
            cluster_center = np.random.uniform(0, 360)
            cluster_size = np.random.randint(10, 30)
            cluster_phases = np.random.normal(cluster_center, 8, cluster_size)
            phases = np.concatenate([phases, cluster_phases])
        
        # 能量也隨機但有偏向低值
        magnitudes = np.random.exponential(qmax_pc * 0.15, len(phases))
        magnitudes = np.clip(magnitudes, 0, qmax_pc * 0.4)
        
        title_suffix = f"(雜訊模式或複雜缺陷 - 檢測到 {len(phases)} 個事件)"
        severity = "待分析"

    # 確保相位在 0-360 之間
    phases = phases % 360

    # 繪製散點圖，點大小根據能量變化
    sizes = 10 + (magnitudes / qmax_pc) * 20  # 動態點大小
    scatter = ax.scatter(phases, magnitudes, c=magnitudes, cmap='plasma', 
                        alpha=0.6, s=sizes, edgecolors='none', label='PD Events')
    
    # 添加色條
    cbar = plt.colorbar(scatter, ax=ax, pad=0.01)
    cbar.set_label('放電強度 (pC)', fontsize=9)
    
    # 標記統計資訊
    stats_text = f"Qmax: {qmax_pc:.0f} pC\n平均: {np.mean(magnitudes):.0f} pC\n事件數: {len(phases)}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    ax.set_title(f"22kV 電纜 PRPD 局部放電相位分析圖 {title_suffix}\n嚴重程度: {severity}", fontsize=12, pad=10)
    ax.set_xlabel("相位角 (Phase Angle, Degree)", fontsize=10)
    ax.set_ylabel("放電量 (Magnitude, pC)", fontsize=10)
    ax.set_xlim(0, 360)
    ax.set_ylim(0, qmax_pc * 1.2)
    ax.grid(True, linestyle=':', alpha=0.4, linewidth=0.5)
    ax.legend(prop={'size': 9}, loc='upper right')
    plt.tight_layout()
    return fig

def generate_didt_plot(fault_type: str):
    """生成 HSCB 直流電流 di/dt 特性曲線 - 使用隨機數據模擬真實情況"""
    # 每次繪圖前確保字體設定正確
    setup_chinese_font()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    t = np.linspace(0, 25, 500) # 增加時間範圍和採樣點

    # 隨機化跳脫門檻值（模擬不同設定）
    trip_threshold = np.random.uniform(7500, 8500)
    
    if fault_type == "short":
        # 短路：極快上升，加入隨機變化
        tau = np.random.uniform(1.5, 2.5)  # 時間常數
        peak_multiplier = np.random.uniform(1.4, 1.7)  # 峰值倍數
        
        # 基礎指數上升曲線
        i_curve = trip_threshold * (1 - np.exp(-t / tau)) * peak_multiplier
        
        # 加入高頻震盪（短路瞬間的電流震盪）
        oscillation = np.random.uniform(100, 300) * np.sin(2 * np.pi * np.random.uniform(5, 10) * t) * np.exp(-t / 3)
        i_curve += oscillation
        
        # 計算實際 di/dt
        dt = t[1] - t[0]
        didt = np.gradient(i_curve, dt)
        max_didt = np.max(didt)
        
        # 隨機決定跳脫時間點
        trip_time = np.random.uniform(2, 4)
        trip_idx = np.argmin(np.abs(t - trip_time))
        
        slope_annotation = f"di/dt max: {max_didt:.0f} A/ms\n(短路特徵)"
        line_color = 'red'
        fault_desc = "高速短路故障"
        
    elif fault_type == "inrush":
        # 湧流：較慢上升，有明顯的二次諧波
        tau = np.random.uniform(7, 10)  # 較慢的時間常數
        peak_multiplier = np.random.uniform(0.75, 0.95)
        
        # 基礎曲線
        i_curve = trip_threshold * (1 - np.exp(-t / tau)) * peak_multiplier
        
        # 加入二次諧波（湧流特徵）
        harmonic2 = np.random.uniform(500, 1000) * np.sin(2 * np.pi * 0.5 * t / 10) * np.exp(-t / 12)
        i_curve += harmonic2
        
        # 加入低頻波動
        ripple = np.random.uniform(200, 400) * np.sin(2 * np.pi * 1.5 * t / 10)
        i_curve += ripple
        
        dt = t[1] - t[0]
        didt = np.gradient(i_curve, dt)
        max_didt = np.max(didt)
        
        trip_time = np.random.uniform(8, 12)
        trip_idx = np.argmin(np.abs(t - trip_time))
        
        slope_annotation = f"di/dt max: {max_didt:.0f} A/ms\n(湧流特徵)"
        line_color = 'orange'
        fault_desc = "列車啟動湧流"
        
    else:  # 其他類型故障
        # 混合模式：中速上升
        tau = np.random.uniform(4, 6)
        peak_multiplier = np.random.uniform(1.0, 1.3)
        
        i_curve = trip_threshold * (1 - np.exp(-t / tau)) * peak_multiplier
        
        # 加入不規則波動（接觸不良等）
        noise_freq = np.random.uniform(3, 8)
        noise = np.random.uniform(300, 600) * np.sin(2 * np.pi * noise_freq * t / 10)
        i_curve += noise
        
        dt = t[1] - t[0]
        didt = np.gradient(i_curve, dt)
        max_didt = np.max(didt)
        
        trip_time = np.random.uniform(5, 8)
        trip_idx = np.argmin(np.abs(t - trip_time))
        
        slope_annotation = f"di/dt max: {max_didt:.0f} A/ms\n(待判定)"
        line_color = 'yellow'
        fault_desc = "未知異常"

    # 加入測量雜訊（模擬真實感測器數據）
    noise = np.random.normal(0, trip_threshold * 0.01, len(t))
    i_curve += noise
    
    # 確保電流非負
    i_curve = np.maximum(i_curve, 0)
    
    # 繪製主電流曲線
    ax.plot(t, i_curve, color=line_color, linewidth=2, label='故障電流波形', alpha=0.9)
    
    # 繪製跳脫門檻線
    ax.axhline(y=trip_threshold, color='cyan', linestyle='--', linewidth=1.5, 
               label=f'76 跳脫門檻 ({trip_threshold:.0f} A)', alpha=0.8)
    
    # 標記跳脫點
    if i_curve[trip_idx] >= trip_threshold:
        ax.plot(t[trip_idx], i_curve[trip_idx], 'ro', markersize=10, 
                label=f'跳脫時刻 ({t[trip_idx]:.1f} ms)', zorder=5)
        ax.axvline(x=t[trip_idx], color='red', linestyle=':', alpha=0.5, linewidth=1)
    
    # 標記最大 di/dt 發生點
    max_didt_idx = np.argmax(np.abs(didt))
    ax.plot(t[max_didt_idx], i_curve[max_didt_idx], 'g^', markersize=10,
            label=f'最大斜率點 ({t[max_didt_idx]:.1f} ms)', zorder=5)
    
    # 添加斜率註解
    ax.annotate(slope_annotation, xy=(t[max_didt_idx], i_curve[max_didt_idx]), 
                xytext=(t[max_didt_idx] + 5, trip_threshold * 0.4),
                arrowprops=dict(facecolor='white', shrink=0.05, width=1.5, headwidth=8), 
                color='white', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
    
    # 統計資訊框
    stats_text = f"故障類型: {fault_desc}\n峰值電流: {np.max(i_curve):.0f} A\n跳脫時間: {t[trip_idx]:.1f} ms\ndi/dt 最大: {max_didt:.0f} A/ms"
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))

    ax.set_title(f"第三軌 HSCB 故障電流 di/dt 特性分析\n分析結果: {fault_desc}", fontsize=12, pad=10)
    ax.set_xlabel("時間 (Time, ms)", fontsize=10)
    ax.set_ylabel("直流電流 (DC Current, A)", fontsize=10)
    ax.set_xlim(0, 25)
    ax.set_ylim(0, np.max(i_curve) * 1.15)
    ax.grid(True, linestyle=':', alpha=0.4, linewidth=0.5)
    ax.legend(prop={'size': 9}, loc='lower right')
    plt.tight_layout()
    return fig

# ==================== 供電環境類別 (含視覺化) ====================

class PowerSupplyEnvironment:
    def __init__(self):
        self.manuals = self._init_manuals()
        self.fault_records = self._init_fault_records()
        # 簡化技術規範，引導 Agent 使用圖表
        self.specs_prompt = """
        技術規範重點：
        1. 電纜 PD：若 Qmax > 500pC，應查看 PRPD 圖譜。內部空隙會在 45/225 度出現群聚。
        2. HSCB 跳脫：需分析電流上升率 (di/dt)。短路斜率極陡，列車啟動湧流斜率較緩。
        """
    
    def _init_manuals(self) -> Dict[str, str]:
        return {
            "22kV電纜絕緣": "SOP-CABLE-05：執行局部放電 (PD) 監測。若發現典型內部放電圖譜 (Void Pattern)，需安排停電更換電纜終端匣。",
            "第三軌高阻抗接地": "SOP-TR-09：調閱 HSCB 故障波形紀錄。若 di/dt 斜率低於設定參數但仍造成跳脫，需巡檢軌道是否有異物造成的間歇性接地。"
        }
    
    def _init_fault_records(self) -> List[Dict]:
        return [
            {"id": "F-2501", "diag": "電纜接頭施工不良", "symptoms": ["PD告警", "Qmax高"], "note": "PRPD圖顯示對稱的 45/225 度群聚訊號。"},
            {"id": "F-2502", "diag": "列車啟動湧流誤跳脫", "symptoms": ["HSCB跳脫", "雨天"], "note": "波形顯示電流上升緩慢，未達短路特徵斜率。"}
        ]

    def handle_tags(self, agent_output: str) -> Optional[str]:
        """解析 Agent 標籤並執行工具調用 (含繪圖)"""
        responses = []
        
        # 處理 lookup 標籤 - 查閱維護手冊
        lookup_matches = re.findall(r"<lookup>(.*?)</lookup>", agent_output, re.DOTALL)
        for keyword in lookup_matches:
            keyword = keyword.strip()
            found = False
            for manual_key, manual_content in self.manuals.items():
                if keyword.lower() in manual_key.lower():
                    responses.append(f"\n<guide> 📖 查閱手冊【{manual_key}】：{manual_content} </guide>\n")
                    found = True
                    break
            if not found:
                responses.append(f"\n<guide> 📖 查無【{keyword}】相關手冊資料 </guide>\n")
        
        # 處理 match 標籤 - 匹配歷史故障記錄
        match_matches = re.findall(r"<match>(.*?)</match>", agent_output, re.DOTALL)
        for query in match_matches:
            query = query.strip().lower()
            matched_records = []
            for record in self.fault_records:
                # 檢查症狀或診斷是否匹配
                if any(query in symptom.lower() for symptom in record['symptoms']) or \
                   query in record['diag'].lower():
                    matched_records.append(record)
            
            if matched_records:
                result = "\n<guide> 🔍 匹配到的歷史案例：\n"
                for rec in matched_records:
                    result += f"  - 案例 {rec['id']}: {rec['diag']}\n"
                    result += f"    症狀: {', '.join(rec['symptoms'])}\n"
                    result += f"    備註: {rec['note']}\n"
                result += "</guide>\n"
                responses.append(result)
            else:
                responses.append(f"\n<guide> 🔍 未找到【{query}】的匹配案例 </guide>\n")
        
        # 處理 search 標籤 - 查詢技術規範
        search_matches = re.findall(r"<search>(.*?)</search>", agent_output, re.DOTALL)
        for search_query in search_matches:
            responses.append(f"\n<literature> 📚 技術規範查詢結果：\n{self.specs_prompt} </literature>\n")
        
        # 處理繪圖請求 (優先處理，直接顯示在 Streamlit)
        plot_match = re.search(r"<plot>(.*?)</plot>", agent_output, re.DOTALL)
        if plot_match:
            plot_params = plot_match.group(1).strip()
            try:
                if "prpd" in plot_params.lower():
                    # 簡易解析參數，實際應用可用更嚴謹的 parser
                    qmax = 600 # 預設或從參數解析
                    if "qmax" in plot_params.lower():
                        qmax_match = re.search(r"qmax[:\s=]*(\d+)", plot_params, re.IGNORECASE)
                        if qmax_match:
                            qmax = float(qmax_match.group(1))
                    pattern = "void" if "void" in plot_params.lower() else "surface"
                    fig = generate_prpd_plot(qmax, pattern)
                    st.pyplot(fig)
                    responses.append("\n<guide> 📊 系統已生成 PRPD 相位分析圖供參 (如上圖)。 </guide>\n")
                elif "didt" in plot_params.lower():
                    fault_type = "short" if "short" in plot_params.lower() else "inrush"
                    fig = generate_didt_plot(fault_type)
                    st.pyplot(fig)
                    responses.append("\n<guide> 📊 系統已生成 HSCB 電流特性曲線圖供參 (如上圖)。 </guide>\n")
            except Exception as e:
                responses.append(f"\n<error> ❌ 圖表生成失敗: {str(e)} </error>\n")

        return "".join(responses) if responses else None

# ==================== 診斷代理流程 ====================

def run_power_agent_loop(fault_data: str, api_key: str):
    """
    執行診斷代理流程
    :param fault_data: 故障描述
    :param api_key: OpenAI API Key (由 UI 傳入)
    """
    env = PowerSupplyEnvironment()
    
    # 在這裡初始化 Client，避免全域變數問題
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"OpenAI Client 初始化失敗: {str(e)}")
        return

    # 更新 System Prompt，教導 Agent 使用完整的工具流程
    system_prompt = """你是一位台北捷運供電廠資深主任工程師。
任務：利用工具診斷供電設備故障原因。

診斷流程（必須依序執行）：
1. 使用 <reason> 分析故障描述，提取關鍵資訊
2. 使用 <lookup> 查閱相關的 SOP 維護手冊
3. 使用 <match> 搜尋類似的歷史故障案例
4. 使用 <search> 查詢技術判斷標準
5. 使用 <plot> 產生視覺化圖表輔助分析（當涉及數值型數據時）
6. 再次使用 <reason> 綜合以上資訊進行推理
7. 使用 <diagnose> 給出最終診斷結論

可用工具標籤：
- <reason>你的思考過程</reason>: 邏輯推理與分析
- <lookup>關鍵字</lookup>: 查閱維護手冊（例如：22kV電纜絕緣、第三軌高阻抗接地）
- <match>症狀關鍵字</match>: 匹配歷史故障記錄（例如：PD告警、HSCB跳脫）
- <search>技術主題</search>: 查詢技術規範與判斷標準
- <plot>參數</plot>: 請求視覺化圖表
    - 參數格式: "prpd, qmax:數值, pattern:void/surface" (電纜局部放電分析)
    - 參數格式: "didt, type:short/inrush" (直流斷路器電流波形)
- <diagnose>最終診斷結論與建議處置</diagnose>: 完成診斷

重要規則：
1. 每次回應只使用一個工具標籤
2. 必須等待系統回饋後再繼續下一步
3. 當遇到數值型描述時（如 PD 值、電流斜率），必須使用 <plot> 工具確認波形特徵
4. 在給出 <diagnose> 之前，必須至少使用過 <lookup>、<match>、<search> 和 <reason>
5. <diagnose> 標籤只能在最後一步使用一次"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"當前故障報告：{fault_data}"}
    ]
    
    for step in range(1, MAX_REASONING_STEPS + 1):
        with st.spinner(f"維修專家思考中 (步驟 {step}/{MAX_REASONING_STEPS})..."):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME, messages=messages, temperature=MODEL_TEMPERATURE
                )
            except Exception as e:
                st.error(f"API 呼叫錯誤 (可能是 Key 無效或配額不足): {str(e)}")
                return

        agent_out = response.choices[0].message.content
        messages.append({"role": "assistant", "content": agent_out})
        
        # 使用 expander 包裹每一步的思考與圖表
        with st.expander(f"🛠️ 步驟 {step}: 專家推理與工具調用", expanded=True):
            # 1. 顯示 Agent 的原始回應
            st.markdown("**🤖 AI 專家輸出：**")
            st.markdown(agent_out)
            
            # 2. 處理標籤並獲取系統回饋
            feedback = env.handle_tags(agent_out)
            
            if feedback:
                st.markdown("**⚙️ 系統回饋：**")
                st.markdown(feedback)
                messages.append({"role": "user", "content": feedback})
            else:
                # 如果沒有工具調用，提示 Agent 繼續
                if "<diagnose>" not in agent_out:
                    hint = "請使用適當的工具標籤繼續診斷流程。"
                    st.info(hint)
                    messages.append({"role": "user", "content": hint})
        
        # 檢查是否完成診斷
        if "<diagnose>" in agent_out:
            diagnose_match = re.search(r"<diagnose>(.*?)</diagnose>", agent_out, re.DOTALL)
            if diagnose_match:
                final_diag = diagnose_match.group(1).strip()
                st.success(f"🎯 診斷完成，最終判定：\n### {final_diag}")
                break
    
    # 如果達到最大步驟數仍未完成
    if step == MAX_REASONING_STEPS:
        st.warning("⚠️ 已達最大推理步驟數，診斷流程終止。請檢查輸入或增加步驟限制。")

# ==================== Streamlit UI ====================

def main():
    st.set_page_config(page_title="TRTC Power-Agent V1", layout="wide", page_icon="⚡")
    st.title("⚡ 台北捷運供電廠智慧電力維護代理系統 V1")
    st.caption("Agentic AI 輔助工具：支援 PRPD 相位圖譜與 di/dt 波形自動分析")

    if 'fault_desc' not in st.session_state: st.session_state.fault_desc = ""

    with st.sidebar:
        st.header("🔑 API 設定")
        api_key = st.text_input("請輸入 OpenAI API Key", type="password")
        if not api_key:
            st.info("⚠️ 請先輸入 API Key 才能執行診斷。")
        
        st.divider()
        
        st.header("📝 故障載入")
        st.markdown("選擇典型案例以測試視覺化功能：")
        if st.button("案例 1: 22kV 電纜高 PD 值告警"):
            st.session_state.fault_desc = "TSS-3 的 22kV 電纜迴路絕緣監測系統發出告警，Qmax 數值達到 650pC，請分析可能原因。"
        if st.button("案例 2: 第三軌 HSCB 跳脫 (疑似湧流)"):
            st.session_state.fault_desc = "早尖峰時段，正線有一部列車啟動時，該區段的 HSCB 發生跳脫，當時天氣晴朗，請協助判斷是否為短路。"
        
        st.divider()
        st.markdown("### 使用說明")
        st.markdown("""
        此版本 Agent 具備**主動繪圖**能力。
        當您輸入的描述包含特定技術特徵（如 PD 值、電流斜率）時，Agent 會在推理步驟中自動產生對應的工程圖表來輔助判斷。
        """)

    fault_input = st.text_area("請輸入故障描述 (或從左側載入案例)：", 
                               value=st.session_state.fault_desc, height=150)

    if st.button("🚀 開始 AI 圖形化診斷分析", type="primary", use_container_width=True):
        if not fault_input.strip():
            st.warning("⚠️ 請先輸入故障描述！")
        elif not api_key:
            st.error("❌ 錯誤：請先在左側欄位輸入 OpenAI API Key！")
        else:
            run_power_agent_loop(fault_input, api_key)

if __name__ == "__main__":
    main()