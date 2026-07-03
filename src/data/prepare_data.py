"""
数据准备脚本
- NER: 从 HuggingFace 缓存提取 ShelterW/chinese_common_ner 真实数据
- 分类: 生成高质量多样化中文新闻数据（10类，每类500条）
"""
import os
import sys
import json
import random
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')


# ============================================================
# Part 1: Processing real NER data from HF cache
# ============================================================

def load_real_ner_data(max_samples=5000):
    """Load the ShelterW/chinese_common_ner dataset from HF cache"""
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

    try:
        from datasets import load_dataset
        print("Loading ShelterW/chinese_common_ner from HF mirror...")
        ds = load_dataset('ShelterW/chinese_common_ner', split='train')
        print(f"  Total available: {len(ds)} samples")

        # Filter: keep sentences with at least one entity
        # Also limit to max_samples for training speed
        data = []
        id2tag = ['O', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER']

        for item in ds:
            tags = item['ner_tags']
            # Keep only sentences with entities
            if any(t != 0 for t in tags):
                tag_strings = [id2tag[t] for t in tags]
                data.append({
                    "tokens": item['tokens'],
                    "tags": tag_strings,
                })
            if len(data) >= max_samples:
                break

        print(f"  Selected {len(data)} sentences with entities")
        return data

    except Exception as e:
        print(f"  Failed to load from HF: {e}")
        print("  Falling back to synthetic NER data...")
        return None


# ============================================================
# Part 2: Generate high-quality classification data
# ============================================================

def generate_classification_data():
    """
    Generate 5000 diverse Chinese news headlines across 10 classes.
    Uses templates + keyword banks + random combinations for diversity.
    """
    categories = {
        "体育": {
            "subjects": ["中国男足", "中国女足", "中国女排", "中国男篮", "中国游泳队", "中国乒乓球队",
                        "中国羽毛球队", "中国田径队", "中国体操队", "中国跳水队", "广东男篮", "辽宁男篮",
                        "北京国安", "上海申花", "广州恒大", "山东鲁能", "武汉三镇", "浙江队",
                        "樊振东", "马龙", "王楚钦", "孙颖莎", "陈梦", "王曼昱", "全红婵", "陈芋汐",
                        "苏炳添", "谢震业", "巩立姣", "刘诗颖", "张雨霏", "覃海洋", "潘展乐",
                        "郑钦文", "朱婷", "李盈莹", "张常宁", "袁心玥", "周琦", "郭艾伦", "赵继伟",
                        "武磊", "吴曦", "张玉宁", "韦世豪", "蒋光太", "颜骏凌", "王大雷"],
            "actions": ["战胜", "击败", "力克", "险胜", "逆转", "横扫", "轻取", "淘汰", "绝杀",
                       "打破纪录", "夺得金牌", "卫冕冠军", "晋级决赛", "获得亚军", "斩获铜牌",
                       "宣布退役", "伤愈复出", "转会加盟", "续约留队", "签约新秀",
                       "创造历史", "追平纪录", "刷新最好成绩", "完成帽子戏法", "上演梅开二度"],
            "objects": ["世界纪录", "亚洲纪录", "全国纪录", "赛季最佳", "个人最好成绩",
                       "世界杯", "世锦赛", "亚运会", "奥运会", "全运会", "中超联赛", "CBA联赛",
                       "温网", "法网", "澳网", "美网", "大满贯", "总冠军", "金牌"],
            "contexts": ["在决赛中", "在半决赛中", "在小组赛中", "在加时赛中", "在点球大战中",
                        "主场作战", "客场挑战", "经过激烈角逐", "凭借出色发挥", "在全场观众的欢呼声中"],
            "details": ["比分3比2", "总比分4比1", "创造个人最佳", "时隔十年再夺冠", "连续三届称霸",
                       "打破尘封20年纪录", "首次闯入决赛", "第六次捧杯", "惊天逆转取胜", "加时绝杀制胜"],
        },

        "财经": {
            "subjects": ["A股市场", "港股市场", "美股市场", "创业板", "科创板", "沪深300",
                        "上证指数", "深证成指", "恒生指数", "纳斯达克", "人民币汇率", "国际油价",
                        "央行", "财政部", "证监会", "银保监会", "国家统计局",
                        "腾讯控股", "阿里巴巴", "字节跳动", "华为技术", "比亚迪", "宁德时代",
                        "茅台股份", "工商银行", "中国平安", "比亚迪", "中芯国际", "隆基绿能",
                        "房地产市场", "新能源汽车", "人工智能产业", "半导体行业", "光伏产业", "医药行业"],
            "actions": ["发布财报", "宣布回购", "获批上市", "完成融资", "股价大涨", "遭遇做空",
                       "宣布降息", "上调利率", "发行国债", "出台新政", "加强监管", "放宽限制",
                       "达成并购", "战略重组", "业务拆分", "跨界布局", "海外扩张"],
            "objects": ["季度报告", "年度财报", "招股说明书", "监管函", "行政处罚决定",
                       "货币政策报告", "国民经济数据", "CPI数据", "PMI指数", "GDP增速",
                       "200亿融资", "50亿回购计划", "千亿市值", "百亿补贴"],
            "contexts": ["在政策利好推动下", "受国际形势影响", "随着经济复苏",
                        "在全球通胀背景下", "随着数字化转型加速", "在新一轮科技革命中"],
            "details": ["涨幅超过5%", "下跌3.2个百分点", "同比增长8.7%", "环比上升0.5%",
                       "市值突破万亿", "成交额创年内新高", "外资持续流入", "主力资金净流入超百亿"],
        },

        "科技": {
            "subjects": ["华为", "苹果", "特斯拉", "OpenAI", "谷歌", "微软", "Meta", "英伟达",
                        "百度", "阿里巴巴", "腾讯", "字节跳动", "小米", "OPPO", "vivo", "荣耀",
                        "科大讯飞", "商汤科技", "旷视科技", "地平线", "寒武纪", "壁仞科技",
                        "中国航天", "SpaceX", "蓝色起源", "NASA", "中国空间站",
                        "中科院", "清华大学", "北京大学", "浙江大学", "上海交大"],
            "actions": ["发布", "推出", "开源", "展示", "测试", "商用", "突破",
                       "研发成功", "实现量产", "获得认证", "申请专利", "达成合作",
                       "首次演示", "完成首飞", "成功发射", "实现超越", "弯道超车"],
            "objects": ["新一代大模型", "GPT-5级别AI", "量子计算机", "自动驾驶系统", "人形机器人",
                       "脑机接口设备", "基因编辑技术", "固态电池", "光子芯片", "超导材料",
                       "载人飞船", "火星探测器", "空间望远镜", "月球基地", "深空探测器",
                       "6G通信技术", "全息显示", "可折叠屏幕", "卫星互联网", "AI芯片"],
            "contexts": ["在全球科技大会上", "在年度开发者大会上", "经过五年秘密研发",
                        "与合作伙伴联合", "在实验室条件下", "通过严格测试后"],
            "details": ["性能提升100倍", "算力达到全球第一", "功耗降低90%", "成本仅为传统方案的1/10",
                       "打破世界纪录", "获得国际大奖", "吸引百万开发者", "开源社区反响热烈"],
        },

        "教育": {
            "subjects": ["教育部", "清华大学", "北京大学", "复旦大学", "浙江大学", "南京大学",
                        "上海交大", "中国科大", "华中科大", "武汉大学", "中山大学", "西安交大",
                        "中小学", "幼儿园", "职业院校", "培训机构", "在线教育平台",
                        "高考", "考研", "公考", "留学申请", "雅思", "托福"],
            "actions": ["发布通知", "出台规定", "启动改革", "公布数据", "调整政策",
                       "扩大招生", "新增专业", "取消考试", "改革方案", "试点推行",
                       "入选双一流", "获评A+学科", "引进顶尖人才", "科研成果转化"],
            "objects": ["新高考方案", "双减政策", "新课标", "学位法", "教师法",
                       "人工智能专业", "集成电路学院", "未来技术学院", "卓越工程师计划"],
            "contexts": ["在新学期开始之际", "在全国教育工作会议上", "经过多年论证",
                        "在广泛征求意见后", "为适应新时代需求"],
            "details": ["录取率创历史新高", "报名人数突破千万", "毕业生就业率达95%",
                       "留学生回国人数增长30%", "考研报名人数首次下降"],
        },

        "军事": {
            "subjects": ["中国海军", "中国空军", "中国陆军", "火箭军", "战略支援部队",
                        "辽宁舰", "山东舰", "福建舰", "055型驱逐舰", "052D型驱逐舰",
                        "歼-20", "歼-35", "运-20", "直-20", "轰-6K", "空警-500",
                        "东风-17", "东风-21D", "东风-26", "东风-41", "巨浪-3",
                        "国防部", "联参部", "各战区", "驻港部队", "驻澳部队"],
            "actions": ["举行演习", "开展训练", "列装部队", "首次亮相", "成功试射",
                       "巡航执法", "护航任务", "维和行动", "人道救援", "撤侨行动",
                       "展示实力", "回应挑衅", "捍卫主权", "例行巡逻", "实战演练"],
            "objects": ["新型驱逐舰", "新一代战斗机", "高超音速导弹", "无人作战平台",
                       "反舰弹道导弹", "舰载电磁炮", "激光武器系统", "量子雷达",
                       "联合军事演习", "海上联演", "空中巡航", "实战化训练"],
            "contexts": ["在南海海域", "在东海防空识别区", "在台湾海峡", "在西太平洋",
                        "在边境地区", "在联合国维和框架下", "根据年度训练计划"],
            "details": ["出动舰机数十架次", "航行通过宫古海峡", "完成实弹射击考核",
                       "精准命中海上目标", "创造多项训练纪录"],
        },

        "游戏": {
            "subjects": ["《王者荣耀》", "《原神》", "《和平精英》", "《崩坏：星穹铁道》",
                        "《英雄联盟》", "《绝地求生》", "《永劫无间》", "《黑神话：悟空》",
                        "《幻塔》", "《鸣潮》", "《明日方舟》", "《崩坏3》", "《蛋仔派对》",
                        "腾讯游戏", "网易游戏", "米哈游", "鹰角网络", "游戏科学",
                        "索尼PlayStation", "微软Xbox", "任天堂Switch", "Steam平台", "Epic"],
            "actions": ["上线新版本", "推出联动活动", "发放福利", "更新赛季", "修复BUG",
                       "全球上线", "登陆新平台", "突破千万销量", "获TGA提名", "夺得年度最佳",
                       "公布新角色", "发布预告片", "开启测试", "正式定档", "加开服务器"],
            "objects": ["新英雄", "新地图", "限定皮肤", "周年庆活动", "电竞赛事",
                       "S14赛季", "2.0大版本", "夏日资料片", "新春活动", "联动限定"],
            "contexts": ["在玩家热切期待中", "经过三年精心打磨", "在上线首日",
                        "在Steam热销榜上", "在全球玩家社区中"],
            "details": ["日活跃用户破亿", "首周流水超10亿", "好评率高达95%",
                       "同时在线人数破百万", "全球用户突破5亿"],
        },

        "娱乐": {
            "subjects": ["春节档", "暑期档", "国庆档", "贺岁档", "五一档",
                        "《流浪地球3》", "《封神第二部》", "《哪吒2》", "《热辣滚烫》", "《第二十条》",
                        "吴京", "沈腾", "贾玲", "黄渤", "王宝强", "刘德华", "周星驰",
                        "张艺谋", "陈凯歌", "郭帆", "饺子", "乌尔善", "宁浩", "徐克",
                        "周杰伦", "林俊杰", "陈奕迅", "邓紫棋", "张杰", "华晨宇", "毛不易",
                        "《乘风破浪》", "《披荆斩棘》", "《歌手》", "《中国好声音》"],
            "actions": ["上映", "开机", "杀青", "定档", "撤档", "路演", "宣传",
                       "发布预告", "公布阵容", "宣布婚讯", "官宣分手", "回应传闻", "发律师函",
                       "开演唱会", "发布新专辑", "参加综艺", "担任导师", "跨界导演"],
            "objects": ["新片", "新歌", "新综艺", "新剧", "纪录片",
                       "票房冠军", "收视第一", "年度最佳", "豆瓣高分", "金鸡奖"],
            "contexts": ["在首映式上", "在杀青宴上", "在综艺节目中", "在社交媒体上",
                        "接受独家采访时", "在粉丝见面会上"],
            "details": ["首日票房破3亿", "累计票房超50亿", "豆瓣评分8.5",
                       "全网播放量破10亿", "演唱会门票3秒售罄"],
        },

        "房产": {
            "subjects": ["北京楼市", "上海楼市", "深圳楼市", "广州楼市", "杭州楼市",
                        "成都楼市", "武汉楼市", "南京楼市", "苏州楼市", "合肥楼市",
                        "万科", "保利", "中海", "华润置地", "招商蛇口", "绿城",
                        "滨江", "建发", "龙湖", "金茂", "越秀", "华发"],
            "actions": ["出台新政", "调整限购", "降低首付", "下调利率", "取消限售",
                       "推出新房源", "打折促销", "开盘售罄", "交付延期", "陷入停工",
                       "发布财报", "债务违约", "资产重组", "引入战投", "退市风险"],
            "objects": ["新盘", "二手房", "学区房", "地铁房", "江景房", "豪宅",
                       "刚需盘", "改善盘", "人才房", "保障房", "共有产权房",
                       "房贷利率", "公积金政策", "契税补贴", "购房资格"],
            "contexts": ["在楼市下行周期中", "在政策暖风频吹下", "随着人口流入",
                        "在轨道交通开通后", "随着产业园区落地"],
            "details": ["均价突破10万每平", "成交量环比上涨30%", "挂牌量创历史新高",
                       "去化周期超24个月", "法拍房数量激增"],
        },

        "时尚": {
            "subjects": ["LV", "Gucci", "Chanel", "Dior", "Hermes", "Prada",
                        "优衣库", "ZARA", "H&M", "UR", "李宁", "安踏", "波司登",
                        "巴黎时装周", "米兰时装周", "纽约时装周", "上海时装周",
                        "美妆品牌", "护肤产品", "香水", "珠宝品牌", "腕表品牌"],
            "actions": ["发布新系列", "宣布代言人", "开设旗舰店", "推出联名款", "涨价",
                       "跨界合作", "签约设计师", "举办大秀", "限时快闪", "限量发售"],
            "objects": ["秋冬系列", "早春系列", "高定系列", "联名胶囊系列",
                       "新款手袋", "限定彩妆", "节日礼盒", "设计师合作款"],
            "contexts": ["在时装周上", "在品牌大秀中", "通过社交媒体官宣",
                        "在线下活动中", "在直播间里"],
            "details": ["上线即售罄", "排队超千人", "小红书种草笔记破万",
                       "二手溢价超3倍", "明星同款秒空"],
        },

        "时政": {
            "subjects": ["习近平主席", "李强总理", "王毅外长", "秦刚国务委员",
                        "全国人大", "全国政协", "国务院", "外交部", "商务部", "国防部",
                        "中国共产党", "中央政府", "香港特区政府", "澳门特区政府",
                        "联合国", "世界卫生组织", "世界银行", "国际货币基金组织",
                        "中俄关系", "中美关系", "中欧关系", "中日关系", "中韩关系", "中印关系"],
            "actions": ["主持召开", "出席", "发表讲话", "签署命令", "通过决议",
                       "严正声明", "强烈谴责", "坚决反对", "表示欢迎", "敦促美方",
                       "达成共识", "签署协议", "深化合作", "访问", "会见", "会谈"],
            "objects": ["重要会议", "国事访问", "联合声明", "白皮书", "政府工作报告",
                       "一带一路倡议", "全球发展倡议", "全球安全倡议", "全球文明倡议",
                       "人类命运共同体", "中国式现代化", "高质量发展"],
            "contexts": ["在联合国大会上", "在G20峰会上", "在金砖国家会议上",
                        "在博鳌亚洲论坛上", "在两会期间", "在记者招待会上"],
            "details": ["取得圆满成功", "达成广泛共识", "签署多项合作协议",
                       "受到国际社会高度评价", "为世界和平发展注入正能量"],
        },
    }

    all_data = []
    for label_idx, (cat_name, config) in enumerate(categories.items()):
        generated = set()
        for _ in range(500):  # 500 per class
            text = None
            attempts = 0
            while text is None or text in generated:
                attempts += 1
                if attempts > 100:
                    break

                # Randomly pick components
                subj = random.choice(config["subjects"])
                act = random.choice(config["actions"])
                obj = random.choice(config["objects"])
                ctx = random.choice(config["contexts"])
                det = random.choice(config["details"])

                # Multiple templates for variety
                templates = [
                    f"{subj}{act}{obj}",
                    f"{subj}{act}{obj}，{det}",
                    f"{ctx}，{subj}{act}{obj}",
                    f"{subj}{act}{obj}，{det}",
                    f"{ctx}，{subj}{act}{obj}，{det}",
                    f"{subj}宣布{act}{obj}",
                    f"{subj}成功{act}{obj}，{det}",
                    f"快讯：{subj}{act}{obj}",
                    f"重磅！{subj}{act}{obj}，{det}",
                    f"独家：{subj}{act}{obj}",
                ]
                text = random.choice(templates)

            if text and text not in generated:
                generated.add(text)
                all_data.append({"text": text, "label": label_idx})

    return all_data


# ============================================================
# Part 3: Save
# ============================================================

def save_splits(data, prefix, output_dir="data/processed"):
    random.seed(42)
    random.shuffle(data)
    n = len(data)
    train_end = int(n * 0.8)
    val_end = train_end + int(n * 0.1)

    splits = {
        f"{prefix}_train.json": data[:train_end],
        f"{prefix}_val.json": data[train_end:val_end],
        f"{prefix}_test.json": data[val_end:],
    }

    os.makedirs(output_dir, exist_ok=True)
    for fname, subset in splits.items():
        fpath = os.path.join(output_dir, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(subset, f, ensure_ascii=False, indent=2)
        print(f"  {fname}: {len(subset)} samples")


def main():
    print("=" * 60)
    print("Semantic-Insight Data Preparation")
    print("=" * 60)

    # 1. NER: real data from HF
    print("\n[1/2] NER Data (ShelterW/chinese_common_ner)")
    ner_data = load_real_ner_data(max_samples=5000)

    if ner_data is None:
        print("  ERROR: Could not load real NER data. Check network.")
        return

    all_tags = []
    for s in ner_data:
        all_tags.extend(s["tags"])
    print(f"  Tag distribution: {dict(Counter(all_tags))}")
    save_splits(ner_data, "ner")

    # 2. Classification: generated
    print("\n[2/2] Classification Data (generated)")
    cls_data = generate_classification_data()
    label_counts = Counter(s["label"] for s in cls_data)
    cat_names = ["体育", "财经", "科技", "教育", "军事", "游戏", "娱乐", "房产", "时尚", "时政"]
    for idx, name in enumerate(cat_names):
        print(f"  {name}: {label_counts.get(idx, 0)}")
    save_splits(cls_data, "news")

    print("\nDone! Ready for training.")


if __name__ == "__main__":
    main()
