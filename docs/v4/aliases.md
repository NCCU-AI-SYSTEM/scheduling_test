# 課名縮寫對照表

資料來源：`data/v4/course_struct.db` 的 `alias` 表，共 815 列。搜尋只用評鑑留言抽出與人工整理、且課程在 114-2 有開的 91 個（使用者 2026/11/01 決定），jieba 自動產生的不用（抽查 27/33 正確）。全部列表在 `data/v4/aliases_all.tsv`。

產生方式：`src/v4/structured/aliases.py`；評鑑留言的證據是提到該縮寫的留言中，屬於這門課的比例。搜尋時的展開：`src/v4/struct_store.py` 的 `expand_aliases()`。

| 縮寫 | 對應課名 | 來源 | 證據 | 說明 |
|---|---|---|---|---|
| 初會 | 初級會計學 | 評鑑留言 | 200 | 200/268 comments mentioning 初會 belong to 初級會計學 |
| 中會 | 中級會計學 | 評鑑留言 | 196 | 196/430 comments mentioning 中會 belong to 中級會計學 |
| 財管 | 財務管理 | 評鑑留言 | 165 | 165/221 comments mentioning 財管 belong to 財務管理 |
| 個經 | 個體經濟學 | 評鑑留言 | 142 | 142/198 comments mentioning 個經 belong to 個體經濟學 |
| 民訴 | 民事訴訟法 | 評鑑留言 | 119 | 119/144 comments mentioning 民訴 belong to 民事訴訟法 |
| 成管 | 成本管理會計 | 評鑑留言 | 91 | 91/94 comments mentioning 成管 belong to 成本管理會計 |
| 總經 | 總體經濟學 | 評鑑留言 | 86 | 86/134 comments mentioning 總經 belong to 總體經濟學 |
| 刑訴 | 刑事訴訟法 | 評鑑留言 | 54 | 54/74 comments mentioning 刑訴 belong to 刑事訴訟法 |
| 高會 | 高級會計學 | 評鑑留言 | 54 | 54/70 comments mentioning 高會 belong to 高級會計學 |
| 數統 | 數理統計學 | 評鑑留言 | 47 | 47/70 comments mentioning 數統 belong to 數理統計學 |
| 國際法 | 國際公法 | 評鑑留言 | 37 | 37/73 comments mentioning 國際法 belong to 國際公法 |
| 成會 | 成本管理會計 | 評鑑留言 | 35 | 35/45 comments mentioning 成會 belong to 成本管理會計 |
| 阿語 | 阿拉伯語 | 評鑑留言 | 35 | 35/69 comments mentioning 阿語 belong to 阿拉伯語 |
| 中憲 | 中華民國憲法及政府 | 評鑑留言 | 34 | 34/44 comments mentioning 中憲 belong to 中華民國憲法及政府 |
| 基影 | 基礎影音製作 | 評鑑留言 | 31 | 31/50 comments mentioning 基影 belong to 基礎影音製作 |
| 民概 | 民法概要 | 評鑑留言 | 31 | 31/38 comments mentioning 民概 belong to 民法概要 |
| 行管 | 行銷管理 | 評鑑留言 | 31 | 31/43 comments mentioning 行管 belong to 行銷管理 |
| 特教 | 特殊教育導論 | 評鑑留言 | 30 | 30/32 comments mentioning 特教 belong to 特殊教育導論 |
| 中韓 | 中級韓國語 | 評鑑留言 | 29 | 29/54 comments mentioning 中韓 belong to 中級韓國語 |
| 線代 | 線性代數 | 評鑑留言 | 29 | 29/42 comments mentioning 線代 belong to 線性代數 |
| 貨銀 | 貨幣銀行學 | 評鑑留言 | 28 | 28/43 comments mentioning 貨銀 belong to 貨幣銀行學 |
| 普心 | 普通心理學 | 評鑑留言 | 25 | 25/34 comments mentioning 普心 belong to 普通心理學 |
| 客語 | 客家語言與文化 | 評鑑留言 | 24 | 24/29 comments mentioning 客語 belong to 客家語言與文化 |
| 微甲 | 微積分甲 | 評鑑留言 | 23 | 23/31 comments mentioning 微甲 belong to 微積分甲 |
| 日會 | 日語會話 | 評鑑留言 | 23 | 23/36 comments mentioning 日會 belong to 日語會話 |
| 寫閱 | 寫作與閱讀 | 評鑑留言 | 21 | 21/24 comments mentioning 寫閱 belong to 寫作與閱讀 |
| 規實 | 規劃實務 | 評鑑留言 | 21 | 21/35 comments mentioning 規實 belong to 規劃實務 |
| 靜影 | 靜態影像設計 | 評鑑留言 | 20 | 20/32 comments mentioning 靜影 belong to 靜態影像設計 |
| 高微 | 高等微積分 | 評鑑留言 | 20 | 20/25 comments mentioning 高微 belong to 高等微積分 |
| 人管 | 人力資源管理 | 評鑑留言 | 19 | 19/23 comments mentioning 人管 belong to 人力資源管理 |
| 心統 | 心理及教育統計學 | 評鑑留言 | 18 | 18/24 comments mentioning 心統 belong to 心理及教育統計學 |
| 客家文 | 客家語言與文化 | 評鑑留言 | 15 | 15/15 comments mentioning 客家文 belong to 客家語言與文化 |
| 政哲 | 政治哲學概論 | 評鑑留言 | 15 | 15/17 comments mentioning 政哲 belong to 政治哲學概論 |
| 傳社 | 傳播與社會 | 評鑑留言 | 14 | 14/17 comments mentioning 傳社 belong to 傳播與社會 |
| 初韓 | 初級韓國語 | 評鑑留言 | 14 | 14/30 comments mentioning 初韓 belong to 初級韓國語 |
| 國公 | 國際公法 | 評鑑留言 | 12 | 12/17 comments mentioning 國公 belong to 國際公法 |
| 社統 | 社會統計 | 評鑑留言 | 12 | 12/24 comments mentioning 社統 belong to 社會統計 |
| 西哲 | 西洋哲學史 | 評鑑留言 | 11 | 11/16 comments mentioning 西哲 belong to 西洋哲學史 |
| 公關企 | 公關策略與企劃 | 評鑑留言 | 10 | 10/10 comments mentioning 公關企 belong to 公關策略與企劃 |
| 比政 | 比較政府與政治 | 評鑑留言 | 10 | 10/15 comments mentioning 比政 belong to 比較政府與政治 |
| 社理 | 社會學理論 | 評鑑留言 | 10 | 10/12 comments mentioning 社理 belong to 社會學理論 |
| 英美導 | 英美法導論 | 評鑑留言 | 10 | 10/11 comments mentioning 英美導 belong to 英美法導論 |
| 法服 | 法律服務 | 評鑑留言 | 9 | 9/11 comments mentioning 法服 belong to 法律服務 |
| 看紀 | 看的方法(紀錄片) | 評鑑留言 | 9 | 9/19 comments mentioning 看紀 belong to 看的方法(紀錄片) |
| 管科 | 管理科學 | 評鑑留言 | 9 | 9/11 comments mentioning 管科 belong to 管理科學 |
| 傳敘 | 傳播敘事 | 評鑑留言 | 8 | 8/8 comments mentioning 傳敘 belong to 傳播敘事 |
| 土經 | 土地經濟學 | 評鑑留言 | 8 | 8/8 comments mentioning 土經 belong to 土地經濟學 |
| 法素 | 法學素養 | 評鑑留言 | 8 | 8/8 comments mentioning 法素 belong to 法學素養 |
| 高韓 | 高級韓國語 | 評鑑留言 | 8 | 8/15 comments mentioning 高韓 belong to 高級韓國語 |
| 傳思 | 傳播與當代思潮 | 評鑑留言 | 7 | 7/9 comments mentioning 傳思 belong to 傳播與當代思潮 |
| 長照 | 長期照護與健康產業發展 | 評鑑留言 | 7 | 7/10 comments mentioning 長照 belong to 長期照護與健康產業發展 |
| 國金 | 國際金融 | 評鑑留言 | 6 | 6/8 comments mentioning 國金 belong to 國際金融 |
| 民調 | 民意與調查 | 評鑑留言 | 6 | 6/10 comments mentioning 民調 belong to 民意與調查 |
| 法倫 | 法律倫理 | 評鑑留言 | 6 | 6/6 comments mentioning 法倫 belong to 法律倫理 |
| 紀採 | 紀實採寫 | 評鑑留言 | 6 | 6/6 comments mentioning 紀採 belong to 紀實採寫 |
| 西方經 | 西方文學經典與人文思維 | 評鑑留言 | 6 | 6/7 comments mentioning 西方經 belong to 西方文學經典與人文思維 |
| 西歐建 | 西歐歷史建築 | 評鑑留言 | 6 | 6/6 comments mentioning 西歐建 belong to 西歐歷史建築 |
| 國組 | 國際組織 | 評鑑留言 | 5 | 5/9 comments mentioning 國組 belong to 國際組織 |
| 戲概 | 戲劇概論 | 評鑑留言 | 5 | 5/6 comments mentioning 戲概 belong to 戲劇概論 |
| 政方 | 政治學方法論 | 評鑑留言 | 5 | 5/10 comments mentioning 政方 belong to 政治學方法論 |
| 遙測 | 遙感探測 | 評鑑留言 | 5 | 5/5 comments mentioning 遙測 belong to 遙感探測 |
| 阿美語 | 阿美族語 | 評鑑留言 | 5 | 5/5 comments mentioning 阿美語 belong to 阿美族語 |
| 電商 | 電子商務實作研習 | 評鑑留言 | 5 | 5/8 comments mentioning 電商 belong to 電子商務實作研習 |
| 中哲 | 中國哲學史 | 評鑑留言 | 4 | 4/8 comments mentioning 中哲 belong to 中國哲學史 |
| 國文字 | 國文-中國語言文字名作選讀 | 評鑑留言 | 4 | 4/6 comments mentioning 國文字 belong to 國文-中國語言文字名作選讀 |
| 數位科 | 數位內容與科技導論 | 評鑑留言 | 4 | 4/7 comments mentioning 數位科 belong to 數位內容與科技導論 |
| 數學邏 | 數學、邏輯與人生 | 評鑑留言 | 4 | 4/9 comments mentioning 數學邏 belong to 數學、邏輯與人生 |
| 日文作 | 日文寫作 | 評鑑留言 | 4 | 4/6 comments mentioning 日文作 belong to 日文寫作 |
| 經思 | 經濟思想史 | 評鑑留言 | 4 | 4/5 comments mentioning 經思 belong to 經濟思想史 |
| 西外 | 西洋外交史 | 評鑑留言 | 4 | 4/7 comments mentioning 西外 belong to 西洋外交史 |
| 貨幣政 | 貨幣理論與政策 | 評鑑留言 | 4 | 4/7 comments mentioning 貨幣政 belong to 貨幣理論與政策 |
| 輔導理 | 輔導原理與實務 | 評鑑留言 | 4 | 4/4 comments mentioning 輔導理 belong to 輔導原理與實務 |
| 初俄 | 初級俄語 | 評鑑留言 | 3 | 3/6 comments mentioning 初俄 belong to 初級俄語 |
| 強執 | 強制執行法與破產法 | 評鑑留言 | 3 | 3/5 comments mentioning 強執 belong to 強制執行法與破產法 |
| 性別文 | 性別與文學理論入門 | 評鑑留言 | 3 | 3/3 comments mentioning 性別文 belong to 性別與文學理論入門 |
| 應阿 | 應用阿拉伯語 | 評鑑留言 | 3 | 3/5 comments mentioning 應阿 belong to 應用阿拉伯語 |
| 所稅 | 所得稅理論與制度 | 評鑑留言 | 3 | 3/5 comments mentioning 所稅 belong to 所得稅理論與制度 |
| 普物 | 普通物理學 | 評鑑留言 | 3 | 3/4 comments mentioning 普物 belong to 普通物理學 |
| 測量實 | 測量學及實習 | 評鑑留言 | 3 | 3/4 comments mentioning 測量實 belong to 測量學及實習 |
| 生活科 | 生活中的生命科學 | 評鑑留言 | 3 | 3/3 comments mentioning 生活科 belong to 生活中的生命科學 |
| 生科 | 生活中的生命科學 | 評鑑留言 | 3 | 3/5 comments mentioning 生科 belong to 生活中的生命科學 |
| 社研 | 社會研究方法 | 評鑑留言 | 3 | 3/6 comments mentioning 社研 belong to 社會研究方法 |
| 行銷理 | 行銷管理 | 評鑑留言 | 3 | 3/5 comments mentioning 行銷理 belong to 行銷管理 |
| 進財 | 進階財務管理 | 評鑑留言 | 3 | 3/5 comments mentioning 進財 belong to 進階財務管理 |
| 野保 | 野生動物保育行銷 | 評鑑留言 | 3 | 3/4 comments mentioning 野保 belong to 野生動物保育行銷 |
| 韓國電 | 韓國轉型正義電影評析 | 評鑑留言 | 3 | 3/6 comments mentioning 韓國電 belong to 韓國轉型正義電影評析 |
| 五管 | 營運與供應鏈管理、行銷管理、人力資源管理、財務管理 | 人工整理 |  | 企管五大管理（生產、行銷、人資、研發、財務）；note 先修五管 5 門、RATE 4 則 |
| 大英 | 大學英文 | 人工整理 |  | RATE 5 則；學生慣用 |
| 經原 | 經濟學 | 人工整理 |  | 政大的課名是 經濟學；RATE 提到 經濟學原理 的 22 則有 19 則在 經濟學 |
| 經濟學原理 | 經濟學 | 人工整理 |  | 同上；forum 查詢 沒修過經濟學原理可以直接修個經嗎 |
| 計程 | 計算機程式設計 | 人工整理 |  | forum 程式設計 or 計程；RATE 11 則 |
