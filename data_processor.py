import streamlit as st
import streamlit.logger
import time
from datetime import datetime
import pytz
from lihkg_api import get_lihkg_topic_list

logger = streamlit.logger.get_logger(__name__)
HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")

async def analyze_lihkg_metadata(user_query, cat_id=1, max_pages=10):
    logger.info(f"開始分析: 分類={cat_id}, 問題='{user_query}'")
    st.session_state.metadata = []
    items = await get_lihkg_topic_list(cat_id, sub_cat_id=0, start_page=1, max_pages=max_pages)
    
    today_start = datetime.now(HONG_KONG_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    today_timestamp = int(today_start.timestamp())
    today_date = today_start.strftime("%Y-%m-%d")
    
    metadata_list = [
        {
            "thread_id": item["thread_id"],
            "title": item["title"],
            "no_of_reply": item["no_of_reply"],
            "last_reply_time": item.get("last_reply_time", ""),
            "like_count": item.get("like_count", 0),
            "dislike_count": item.get("dislike_count", 0),
        }
        for item in items
    ]
    
    if "列出" in user_query and "所有" in user_query and "標題" in user_query:
        logger.info(f"檢測到列出所有標題請求，跳過篩選和排序，直接列出所有帖子標題")
        st.session_state.metadata = metadata_list
        titles = [f"- {item['title']}" for item in metadata_list]
        return f"以下是分類 {cat_id} 抓取到的所有帖子標題（共 {len(titles)} 篇）：\n" + "\n".join(titles)
    
    total_items = len(metadata_list)
    min_replies = 125
    
    reply_filtered = [item for item in metadata_list if item["no_of_reply"] >= min_replies]
    reply_filtered_out = total_items - len(reply_filtered)
    
    time_filtered = [
        item for item in reply_filtered
        if int(item["last_reply_time"]) >= today_timestamp
    ]
    time_filtered_out = len(reply_filtered) - len(time_filtered)
    
    filtered_items = time_filtered
    
    filter_report = (
        f"篩選報告：\n"
        f"- 總共抓取 {total_items} 篇帖子\n"
        f"- 條件 1：回覆數 ≥ {min_replies}，過濾掉 {reply_filtered_out} 篇（剩餘 {len(reply_filtered)} 篇）\n"
        f"- 條件 2：最後回覆時間在 {today_date}，過濾掉 {time_filtered_out} 篇\n"
        f"- 最終符合條件的帖子數：{len(filtered_items)} 篇\n"
    )
    if len(filtered_items) < 10:
        filter_report += "建議：若需更多帖子，可降低回覆數門檻（例如 ≥ 50）或移除時間限制。\n"
    
    logger.info(
        f"篩選詳情: cat_id={cat_id}\n"
        f"- 總帖子數={total_items}\n"
        f"- 因回覆數 < {min_replies} 被過濾: {reply_filtered_out} 篇\n"
        f"- 因最後回覆時間不在 {today_date} 被過濾: {time_filtered_out} 篇\n"
        f"- 最終符合條件: {len(filtered_items)} 篇"
    )
    logger.info(f"過濾後帖子數: cat_id={cat_id}, 符合條件數={len(filtered_items)}")
    
    if not filtered_items:
        cat_name = {1: "吹水台", 2: "熱門台", 5: "時事台", 14: "上班台", 15: "財經台", 29: "成人台", 31: "創意台"}.get(cat_id, "未知分類")
        return (
            f"今日 {cat_name} 無符合條件的帖子（總回覆數 ≥ {min_replies} 且最後回覆在今日），建議查看熱門台（cat_id=2）。\n\n"
            f"{filter_report}"
        )
    
    metadata_text = "\n".join([
        f"帖子 ID: {item['thread_id']}, 標題: {item['title']}, 回覆數: {item['no_of_reply']}, 最後回覆: {item['last_reply_time']}, 正評: {item['like_count']}, 負評: {item['dislike_count']}"
        for item in filtered_items
    ])
    
    cat_name = {1: "吹水台", 2: "熱門台", 5: "時事台", 14: "上班台", 15: "財經台", 29: "成人台", 31: "創意台"}.get(cat_id, "未知分類")
    prompt = f"""
使用者問題：{user_query}

以下是 LIHKG 討論區今日（2025-04-16）篩選出的帖子元數據，分類為 {cat_name}（cat_id={cat_id}），條件為總回覆數 ≥ {min_replies} 且最後回覆在今日：
{metadata_text}

以繁體中文回答，基於帖子標題和正負評數量，執行以下步驟：
1. 解析問題意圖，識別核心主題（如財經、情緒、搞笑、爭議、時事、生活等）。
2. 根據問題主題和分類特性，排序帖子：
   - 優先考慮標題與問題的相關性。
   - 根據分類調整正負評的權重：
     - 吹水台（cat_id=1）、時事台（cat_id=5）：正負評高的帖子（正評或負評 ≥ 50）可能更有趣或更有立場，應提高優先級。
     - 財經台（cat_id=15）：正負評不應過分影響排序，應更注重標題的資訊性。
     - 其他分類（如熱門台、上班台、成人台、創意台）：正負評可作為次要參考，適度提升有趣或爭議性帖子的優先級。
3. 返回與問題相關的帖子 ID 列表（每行一個數字），不相關的帖子不應包含在列表中。若無相關帖子，返回空列表。

輸出格式：
- 僅返回相關帖子 ID（每行一個數字），無其他內容。
"""
    
    call_id = f"sort_{cat_id}_{int(time.time())}"
    response = "".join(list(async_to_sync_stream(stream_grok3_response(prompt, call_id=call_id))))
    
    thread_ids = re.findall(r'\b(\d{5,})\b', response, re.MULTILINE)
    valid_ids = [str(item["thread_id"]) for item in filtered_items]
    sorted_ids = [tid for tid in thread_ids if tid in valid_ids]
    
    logger.info(f"AI 排序後帖子: 提取={thread_ids}, 有效={sorted_ids}")
    
    if not sorted_ids:
        logger.warning(f"無相關帖子: 分類={cat_id}")
        return (
            f"今日 {cat_name} 無與問題相關的帖子，建議查看熱門台（cat_id=2）。\n\n"
            f"{filter_report}"
        )
    
    st.session_state.metadata = []
    for tid in sorted_ids:
        for item in filtered_items:
            if str(item["thread_id"]) == tid:
                st.session_state.metadata.append(item)
                break
    
    metadata_text = "\n".join([
        f"帖子 ID: {item['thread_id']}, 標題: {item['title']}, 回覆數: {item['no_of_reply']}, 最後回覆: {item['last_reply_time']}, 正評: {item['like_count']}, 負評: {item['dislike_count']}"
        for item in st.session_state.metadata
    ])
    
    prompt = f"""
使用者問題：{user_query}

以下是 LIHKG 討論區今日（2025-04-16）篩選並排序後的帖子元數據，分類為 {cat_name}（cat_id={cat_id}），已按相關性排序：
{metadata_text}

以繁體中文回答，基於所有帖子標題，綜合分析討論區的廣泛意見，直接回答問題，禁止生成無關內容。執行以下步驟：
1. 解析問題意圖，識別核心主題（如財經、情緒、搞笑、爭議、時事、生活等）。
2. 總結網民整體觀點（100-150 字），提取標題關鍵詞，直接回答問題，適配分類語氣：
   - 吹水台（cat_id=1）：輕鬆，提取搞笑、荒誕話題。
   - 熱門台（cat_id=2）：聚焦高熱度討論。
   - 時事台（cat_id=5）：關注爭議、事件。
   - 上班台（cat_id=14）：聚焦職場、生活。
   - 財經台（cat_id=15）：偏市場、投資。
   - 成人台（cat_id=29）：適度處理敏感話題。
   - 創意台（cat_id=31）：注重趣味、創意。
3. 若標題不足以詳細回答，註明：「可進一步分析帖子內容以提供補充細節。」
4. 若無相關帖子，說明：「今日 {cat_name} 無符合問題的帖子，建議查看熱門台（cat_id=2）。」

輸出格式：
- 總結：100-150 字，直接回答問題，概述網民整體觀點，說明依據（如標題關鍵詞）。
- 篩選報告：
{filter_report}
"""
    return prompt
