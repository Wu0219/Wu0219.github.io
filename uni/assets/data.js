/* ===========================================================================
 * data.js —— 大学生活性价比计算器 · 纯数据层
 * ---------------------------------------------------------------------------
 * 所有系数都是主观权重，不是统计拟合。不认同就改这个文件，改完刷新即可。
 *
 * 适用范围：本科生 / 授课型硕士（Taught Master）
 *   不适用于博士与研究型硕士 —— 那类项目有津贴、有实验室作息、成本结构
 *   与生活形态都不一样，硬套会严重失真。
 *
 * 系数约定：所有 v 值以 1.00 为「当地中位水平」，>1 更好，<1 更差。
 * =========================================================================== */

(function (root) {
  'use strict';

  /* ---------------------------------------------------------------------
   * 全局常量
   * ------------------------------------------------------------------- */
  var GLOBAL = {
    // PPP 锚：把各国货币折算成「中国等效人民币购买力」，用于跨国横向比较。
    // 与 job_calculator 使用同一套锚点，保证两个工具口径一致。
    PPP_ANCHOR: 4.19,
    MONTHS_PER_YEAR: 10,   // 学年按 10 个月计（寒暑假不在校）

    /* ── 成本处理（v2，重构过）────────────────────────────────────
     * v1 用「比例地板 0.60」做除零保护，实测四个实习场景全部触底：
     * 月入 1299 与 7794（差 6 倍）在成本维度上完全等价，分辨率归零，
     * 而且一份 6 小时/周的家教就能让得分跳两个评级档。v2 改三件事：
     *   1) 绝对地板：净支出不低于「基准 × RIGID_RATIO」。食物、通讯、
     *      日用这些刚性开支不可能被实习收入抵扣掉，用它做除零保护比
     *      比例地板更有物理意义，饱和点也推得更远。
     *   2) 两端同时做幂压缩：v1 省钱端最多 1.67 倍，超支端却能无限逼近
     *      0（实测出现 0.118），惩罚严重不对称。
     *   3) 多赚的钱改对数加成，上限 12% → 45%。否则「收入通道」全是低顶、
     *      「工时通道」却能自由跑满 −40%，模型会系统性否定「用时间换钱」。
     */
    RIGID_RATIO: 0.45,        // 刚性开支占当地基准的比例 = 净支出绝对地板
    COST_EXP: 0.75,           // 成本比压缩指数，两端同时收敛
    COST_MAX: 2.60,           // 成本比上限
    // 两端在对数尺度上大致对称：省钱端最多 1/0.45^0.75 ≈ 1.82 倍，
    // 超支端最多 1/2.60^0.75 ≈ 0.49 倍，不再是「奖励封顶、惩罚敞口」。
    // 0.30 试过，触底样本会冲到 3.6~4.5，成本一项仍然盖过其余所有维度。
    SURPLUS_COEF: 0.16,       // 净赚加成 = 1 + min(MAX, COEF × ln(1+超出/基准))
    SURPLUS_MAX_BONUS: 0.45,

    /* 四个质量系数直接连乘会四重复合：各偏离 15% 时结果偏离 75%，
     * 实测极端样本跨度 26 倍，中位数被推到 1.4（锚点本应是 1.0）。
     * 统一加压缩指数，保持排序不变但把跨度收回可解释范围。 */
    QUALITY_EXP: 0.75
  };

  /* ---------------------------------------------------------------------
   * 国家 / 地区
   *   ppp        本币 / 国际元（越小说明本币越「值钱」）
   *   regions    该国的城市层级，含月度基准花销（本币）
   *   baseLiving  月生活费基准（吃饭 + 日常，不含住宿）
   *   baseHousing 月住宿基准 —— **学校宿舍口径**
   *   baseRent    月住宿基准 —— **校外合租单间的当地市场行情**
   *
   * 为什么要两个住宿基准：表单直接问「你每月住宿花多少」，如果一律拿宿舍价
   * 去比，租房的学生必然显得「花超了」—— 实测上海那位住独卫双人间的学生
   * 因此被误判为「略低于当地水平」。基准必须跟着居住形式切换，
   * 住宿舍就比宿舍价，租房就比租房行情，这样比的才是同一类东西。
   * ------------------------------------------------------------------- */
  var COUNTRIES = [
    {
      key: 'CN', label: '中国大陆', cur: '¥', ppp: 4.19,
      regions: [
        { key: 'tier1c', label: '一线核心城区（北上广深 市区）', baseLiving: 2000, baseHousing: 260, baseRent: 2800 },
        { key: 'tier1',  label: '一线城市（北上广深）',           baseLiving: 1800, baseHousing: 220, baseRent: 2200 },
        { key: 'newt1',  label: '新一线（杭州·南京·成都·武汉·西安等）', baseLiving: 1550, baseHousing: 180, baseRent: 1500 },
        { key: 'tier2',  label: '二线省会 / 计划单列市',          baseLiving: 1350, baseHousing: 150, baseRent: 1200 },
        { key: 'tier3',  label: '三线及以下地级市',               baseLiving: 1150, baseHousing: 120, baseRent: 900 },
        { key: 'county', label: '县城 / 独立校区小镇',            baseLiving: 1000, baseHousing: 100, baseRent: 700 }
      ]
    },
    {
      // 香港住宿基准上调约 90%：HK$4500 对应的是「抽中的补贴校内宿舍」，
      // 属于稀缺供给，多数授课型硕士拿不到。校外合租单间市场价 6000-9000。
      key: 'HK', label: '中国香港', cur: 'HK$', ppp: 5.86,
      regions: [
        { key: 'hkurban', label: '港岛 / 九龙市区', baseLiving: 5500, baseHousing: 4500, baseRent: 8500 },
        { key: 'hknt',    label: '新界 / 近郊校区', baseLiving: 4800, baseHousing: 3800, baseRent: 6700 }
      ]
    },
    {
      key: 'ES', label: '西班牙', cur: '€', ppp: 0.62,
      regions: [
        // 房租按 HousingAnywhere 2025 Q3 单间均价校准（巴塞 €650 / 马德里 €620）
        // 西班牙的 residencia 通常比 piso compartido 更贵，与国内相反
        { key: 'bcnmad', label: '巴塞罗那 / 马德里',            baseLiving: 480, baseHousing: 750, baseRent: 620 },
        { key: 'esmid',  label: '瓦伦西亚 · 塞维利亚 · 毕尔巴鄂等', baseLiving: 370, baseHousing: 520, baseRent: 420 },
        { key: 'essmall',label: '小城市 / 大学城（萨拉曼卡等）',   baseLiving: 300, baseHousing: 420, baseRent: 340 }
      ]
    },
    {
      key: 'EU', label: '其他欧元区', cur: '€', ppp: 0.72,
      regions: [
        { key: 'eucap',  label: '首都 / 一线（巴黎·柏林·阿姆斯特丹等）', baseLiving: 520, baseHousing: 750, baseRent: 700 },
        { key: 'eumid',  label: '中型城市',        baseLiving: 430, baseHousing: 520, baseRent: 480 },
        { key: 'eusmall',label: '小城 / 大学城',   baseLiving: 370, baseHousing: 420, baseRent: 380 }
      ]
    },
    {
      key: 'UK', label: '英国', cur: '£', ppp: 0.70,
      regions: [
        // 按 Save the Student 2025 全国学生调查校准：全伦敦总花费均值 £1269，
        // 均租 £812。v1 的 £1450 裸基准已高于实际均值，再乘市中心系数会失真。
        { key: 'london', label: '伦敦',              baseLiving: 500, baseHousing: 950, baseRent: 820 },
        { key: 'ukbig',  label: '曼城·伯明翰·爱丁堡等', baseLiving: 400, baseHousing: 620, baseRent: 550 },
        { key: 'uksmall',label: '小城 / 大学城',      baseLiving: 360, baseHousing: 500, baseRent: 450 }
      ]
    },
    {
      key: 'US', label: '美国', cur: '$', ppp: 1.00,
      regions: [
        { key: 'usmajor', label: '纽约·湾区·波士顿等', baseLiving: 950, baseHousing: 1500, baseRent: 1800 },
        { key: 'usmid',   label: '中型城市',          baseLiving: 750, baseHousing: 1000, baseRent: 1100 },
        { key: 'ustown',  label: '大学城 / 小镇',      baseLiving: 650, baseHousing: 850,  baseRent: 850 }
      ]
    },
    {
      key: 'AU', label: '澳大利亚', cur: 'A$', ppp: 1.47,
      regions: [
        { key: 'ausyd', label: '悉尼 / 墨尔本',   baseLiving: 1100, baseHousing: 1500, baseRent: 1400 },
        { key: 'aumid', label: '布里斯班·珀斯·阿德莱德', baseLiving: 900, baseHousing: 1100, baseRent: 1000 },
        { key: 'autown',label: '小城 / 大学城',   baseLiving: 800, baseHousing: 900,  baseRent: 800 }
      ]
    },
    {
      key: 'JP', label: '日本', cur: '¥JP', ppp: 92,
      regions: [
        { key: 'jptokyo', label: '东京 23 区',        baseLiving: 70000, baseHousing: 62000, baseRent: 80000 },
        { key: 'jpbig',   label: '大阪·京都·名古屋等', baseLiving: 58000, baseHousing: 45000, baseRent: 55000 },
        { key: 'jpsmall', label: '地方城市',          baseLiving: 50000, baseHousing: 35000, baseRent: 42000 }
      ]
    },
    {
      key: 'KR', label: '韩国', cur: '₩', ppp: 861,
      regions: [
        { key: 'krseoul', label: '首尔',     baseLiving: 700000, baseHousing: 450000, baseRent: 600000 },
        { key: 'krbig',   label: '釜山·大邱等', baseLiving: 580000, baseHousing: 340000, baseRent: 420000 },
        { key: 'krsmall', label: '地方城市',  baseLiving: 500000, baseHousing: 280000, baseRent: 330000 }
      ]
    },
    {
      key: 'SG', label: '新加坡', cur: 'S$', ppp: 0.84,
      regions: [
        // 校内宿舍 S$900 是稀缺补贴价，硕士生第二年基本要校外租房
        { key: 'sg', label: '新加坡（全岛）', baseLiving: 750, baseHousing: 900, baseRent: 1300 }
      ]
    },
    {
      key: 'MY', label: '马来西亚', cur: 'RM', ppp: 1.75,
      regions: [
        { key: 'mykl',    label: '吉隆坡',   baseLiving: 1400, baseHousing: 900, baseRent: 1100 },
        { key: 'mysmall', label: '其他城市', baseLiving: 1100, baseHousing: 600, baseRent: 750 }
      ]
    },
    {
      key: 'TH', label: '泰国', cur: '฿', ppp: 12.4,
      regions: [
        // 审查确认这一档基准合理，未作调整
        { key: 'thbkk',   label: '曼谷',     baseLiving: 14000, baseHousing: 7000, baseRent: 9000 },
        { key: 'thsmall', label: '其他城市', baseLiving: 11000, baseHousing: 5000, baseRent: 6000 }
      ]
    }
  ];

  /* ---------------------------------------------------------------------
   * 校区地段
   *   v       生活便利度（进分子）
   *   costIdx 该地段对基准花销的抬升（进分母的基准，不是你的实际花销）
   * ------------------------------------------------------------------- */
  /* 地段对基准花销的影响必须拆开：
   * 房租的市中心溢价是真实的（伦敦 Zone1 vs 全市均价约 1.4 倍），
   * 但地铁月票和超市价格并不会因为住市中心就贵 40% ——
   * v1 把同一个系数乘在「生活费 + 住宿」整体上，是伦敦基准虚高
   * £2059（真实中位约 £1700）的直接原因。
   *   houseIdx 只作用于住宿，livingIdx 只作用于生活费且幅度很小。 */
  var LOCATIONS = [
    { key: 'core',    v: 1.16, houseIdx: 1.42, livingIdx: 1.08, label: '市中心核心区',
      hint: '出门就是地铁商圈，实习面试通勤成本最低；代价是房租顶格' },
    { key: 'urban',   v: 1.09, houseIdx: 1.18, livingIdx: 1.03, label: '市区（非核心）',
      hint: '在城市建成区内，生活配套完整，进市中心半小时以内' },
    { key: 'fringe',  v: 1.00, houseIdx: 1.00, livingIdx: 1.00, label: '城郊结合部',
      hint: '基准档。校门口有基本商业，进城需要专门安排一趟' },
    { key: 'unitown', v: 0.94, houseIdx: 0.90, livingIdx: 0.97, label: '大学城 / 高教园区',
      hint: '几所学校扎堆，学生生态好但只有学生生态，物价低' },
    { key: 'remote',  v: 0.82, houseIdx: 0.80, livingIdx: 0.95, label: '远郊独立校区',
      hint: '进城单程一小时以上，实习和兼职机会被地理位置直接卡死' }
  ];

  var COMMUTE = [
    { v: 1.06, label: '地铁 / 轻轨直达', hint: '出行不需要提前规划' },
    { v: 1.00, label: '公交 30 分钟内可进城' },
    { v: 0.95, label: '有校车，但班次固定' },
    { v: 0.88, label: '公交需 1 小时以上' },
    { v: 0.80, label: '基本得打车 / 自己有车才方便' }
  ];

  /* ---------------------------------------------------------------------
   * 住宿维度（加权平均，中位 = 1.00）
   * ------------------------------------------------------------------- */
  var DORM_DIMS = [
    {
      key: 'housing', label: '居住形式', weight: 0.16,
      hint: '自由度与成本的主要分水岭',
      options: [
        { v: 1.22, label: '校外整租一居 / 单身公寓', hint: '完全自由，代价是这一项的花销会明显拉高分母' },
        { v: 1.12, label: '校外合租（有自己的卧室）', hint: '性价比常见最优解' },
        { v: 1.02, label: '校内公寓式宿舍',          hint: '带独立生活单元的新式宿舍' },
        { v: 1.00, label: '普通校内宿舍',            hint: '基准档' },
        { v: 0.90, label: '校外统一安排的宿舍',      hint: '既没有校内的便利，也没有租房的自由' },
        { v: 0.84, label: '走读 / 住家里',           hint: '省钱，但错过大部分校园社交' }
      ]
    },
    {
      key: 'roomSize', label: '房间住几个人', weight: 0.20,
      hint: '对睡眠质量和心理空间影响最大的单项',
      options: [
        { v: 1.30, label: '单人间',   hint: '独处空间是刚需，不是奢侈' },
        { v: 1.14, label: '双人间' },
        { v: 1.00, label: '三人间',   hint: '基准档' },
        { v: 0.92, label: '四人间',   hint: '国内最常见配置' },
        { v: 0.78, label: '六人间' },
        { v: 0.62, label: '八人及以上', hint: '作息一旦不同步，几乎无法自我调节' }
      ]
    },
    {
      key: 'bathroom', label: '卫生间', weight: 0.14,
      options: [
        { v: 1.22, label: '房间内独立卫生间' },
        { v: 1.08, label: '宿舍单元内共用（2-4 人）' },
        { v: 1.00, label: '楼层公共卫生间',  hint: '基准档' },
        { v: 0.84, label: '楼层公共且数量紧张', hint: '早高峰要排队' },
        { v: 0.70, label: '需要出楼 / 旱厕' }
      ]
    },
    {
      key: 'shower', label: '洗澡条件', weight: 0.14,
      hint: '这一项在冬天的权重远比想象中高',
      options: [
        { v: 1.24, label: '房间内独立淋浴，24 小时热水' },
        { v: 1.10, label: '宿舍单元内共用淋浴' },
        { v: 1.00, label: '楼层公共浴室，热水时段较长', hint: '基准档' },
        { v: 0.82, label: '公共澡堂，定时开放' },
        { v: 0.66, label: '需走出宿舍楼去澡堂，且限时' }
      ]
    },
    {
      key: 'power', label: '断电断网', weight: 0.13,
      options: [
        { v: 1.18, label: '24 小时通电通网，不限功率' },
        { v: 1.06, label: '不断电，但限大功率电器' },
        { v: 1.00, label: '深夜断电（23:30 后）',  hint: '基准档' },
        { v: 0.86, label: '断电且断网（23:00 前后）' },
        { v: 0.72, label: '断得早（22:30 前）或经常计划外停电' }
      ]
    },
    {
      key: 'roommate', label: '舍友关系', weight: 0.17,
      hint: '室友是唯一你无法用钱解决、又天天面对的变量',
      options: [
        { v: 1.32, label: '作息一致，相处融洽，能一起吃饭出去玩' },
        { v: 1.14, label: '基本和谐，互相尊重' },
        { v: 1.00, label: '各过各的，没什么交集也没什么矛盾', hint: '基准档' },
        { v: 0.80, label: '有明显摩擦（作息、卫生、外放声音）' },
        { v: 0.58, label: '长期冲突，回宿舍会有心理负担' },
        { v: 0.95, label: '单人住，没有这个变量' }
      ]
    },
    {
      key: 'climate', label: '空调 / 供暖', weight: 0.06,
      hint: '南方的夏天和北方的冬天，这一项直接决定能不能正常学习',
      options: [
        { v: 1.16, label: '独立空调 + 供暖（或全年恒温）' },
        { v: 1.06, label: '有独立空调，冬天够用' },
        { v: 1.00, label: '中央空调定时供应', hint: '基准档' },
        { v: 0.84, label: '只有风扇 / 只有暖气' },
        { v: 0.68, label: '什么都没有' }
      ]
    }
  ];

  /* 居住形式 → 该用哪个住宿基准去比。
   * 下标与 DORM_DIMS.housing 的选项一一对应。
   *   base: 'rent' 用当地租房行情，'dorm' 用宿舍价
   *   k:    在该基准上的倍数（整租比合租贵、公寓式宿舍比普通宿舍贵）*/
  var HOUSING_BASE = [
    { base: 'rent', k: 1.55 },   // 校外整租一居 / 单身公寓
    { base: 'rent', k: 1.00 },   // 校外合租
    { base: 'dorm', k: 1.60 },   // 校内公寓式宿舍
    { base: 'dorm', k: 1.00 },   // 普通校内宿舍
    { base: 'dorm', k: 1.35 },   // 校外统一安排的宿舍
    { base: 'dorm', k: 0.15 }    // 走读 / 住家里
  ];

  /* ---------------------------------------------------------------------
   * 校园生活维度
   * ------------------------------------------------------------------- */
  var CAMPUS_DIMS = [
    {
      key: 'curfew', label: '门禁', weight: 0.20,
      hint: '门禁时间直接决定你能不能做兼职、实习、参加校外活动',
      options: [
        { v: 1.18, label: '无门禁，刷卡随时进出' },
        { v: 1.06, label: '24:00 后门禁，可登记进入' },
        { v: 1.00, label: '23:00 门禁', hint: '基准档' },
        { v: 0.86, label: '22:30 门禁，晚归要说明' },
        { v: 0.70, label: '22:00 前门禁 + 严格查寝' }
      ]
    },
    {
      key: 'strictness', label: '教学管理强度', weight: 0.18,
      options: [
        { v: 1.16, label: '完全自主，只看考核结果' },
        { v: 1.06, label: '正常点名，但时间自己安排' },
        { v: 1.00, label: '有考勤要求，强度适中', hint: '基准档' },
        { v: 0.86, label: '强制早操 / 晚自习其一' },
        { v: 0.72, label: '早操 + 晚自习 + 频繁查寝，接近高中' }
      ]
    },
    {
      key: 'canteen', label: '食堂', weight: 0.16,
      hint: '一天三顿，四年一万两千顿',
      options: [
        { v: 1.18, label: '便宜好吃，多个食堂可选' },
        { v: 1.06, label: '正常水平，吃得下去' },
        { v: 1.00, label: '一般，偶尔要点外卖', hint: '基准档' },
        { v: 0.86, label: '难吃或贵，主要靠外卖' },
        { v: 0.74, label: '选择极少，且校外没什么餐饮' }
      ]
    },
    {
      key: 'facility', label: '校园设施', weight: 0.16,
      hint: '图书馆座位、自习室、体育馆、实验设备',
      options: [
        { v: 1.16, label: '设施新且充足，从不用抢' },
        { v: 1.06, label: '基本够用，考试周紧张' },
        { v: 1.00, label: '一般，需要提前占座', hint: '基准档' },
        { v: 0.86, label: '明显不足，长期靠抢' },
        { v: 0.74, label: '老旧或缺失' }
      ]
    },
    {
      key: 'social', label: '社团与社交氛围', weight: 0.16,
      options: [
        { v: 1.18, label: '活动丰富，很容易找到同类' },
        { v: 1.06, label: '有一些，参与感还行' },
        { v: 1.00, label: '一般，看个人主动性', hint: '基准档' },
        { v: 0.88, label: '比较冷清' },
        { v: 0.76, label: '几乎没有，人际关系仅限班级' }
      ]
    },
    {
      key: 'safety', label: '治安与心理支持', weight: 0.14,
      options: [
        { v: 1.14, label: '安全，且有靠谱的心理咨询服务' },
        { v: 1.04, label: '安全，服务一般' },
        { v: 1.00, label: '没什么问题', hint: '基准档' },
        { v: 0.88, label: '偶有治安事件 / 求助无门' },
        { v: 0.74, label: '让人不安' }
      ]
    }
  ];

  /* ---------------------------------------------------------------------
   * 发展前景维度
   * ------------------------------------------------------------------- */
  var PROSPECT_DIMS = [
    {
      key: 'major', label: '专业前景', weight: 0.34,
      hint: '按毕业 3-5 年内的就业面与议价能力粗分档，不代表学科价值',
      options: [
        { v: 1.30, label: '计算机 / 人工智能 / 数据科学' },
        { v: 1.24, label: '临床医学 / 口腔（长周期高门槛）' },
        { v: 1.18, label: '电子信息 / 集成电路 / 自动化' },
        { v: 1.14, label: '金融 / 会计 / 精算' },
        { v: 1.10, label: '电气 / 机械 / 材料等主干工科' },
        { v: 1.06, label: '法学 / 药学 / 护理' },
        { v: 1.02, label: '师范 / 教育' },
        { v: 1.00, label: '经管 / 市场营销 / 物流', hint: '基准档' },
        { v: 0.96, label: '外语 / 翻译 / 国际关系' },
        { v: 0.94, label: '数学 / 物理 / 化学等基础理科', hint: '本科直接就业偏窄，读研后显著改善' },
        { v: 0.90, label: '新闻传播 / 中文 / 社会学' },
        { v: 0.88, label: '设计 / 建筑 / 艺术' },
        { v: 0.84, label: '历史 / 哲学 / 人类学' },
        { v: 0.82, label: '农林 / 生物 / 环境', hint: '俗称「四大天坑」，本科就业口径确实窄' },
        { v: 0.86, label: '旅游 / 酒店 / 体育' }
      ]
    },
    /* 院校层次这一维已移除。
     * 它对在校生完全不可改变，填了也只是把一个既定事实换算成分数，
     * 而「学长学姐的实际去向」这一维已经把院校的真实价值反映进来了 ——
     * 后者是结果，前者只是标签。移除后其余四项按比例上调。 */
    {
      key: 'localJobs', label: '本地就业机会', weight: 0.24,
      hint: '学校所在地有没有你这个专业的对口岗位，直接决定实习成本',
      options: [
        { v: 1.28, label: '本地就是该行业的核心城市', hint: '实习和秋招都不用出远门' },
        { v: 1.14, label: '本地有成规模的对口产业' },
        { v: 1.00, label: '本地有一些机会，但选择有限', hint: '基准档' },
        { v: 0.86, label: '基本要去邻近大城市找' },
        { v: 0.72, label: '本地几乎没有对口岗位', hint: '每次面试都是一次长途差旅' }
      ]
    },
    {
      key: 'internship', label: '实习可得性', weight: 0.18,
      options: [
        { v: 1.22, label: '大二起就能边上课边实习' },
        { v: 1.10, label: '课程安排允许，通勤可接受' },
        { v: 1.00, label: '需要请假或压缩课程', hint: '基准档' },
        { v: 0.86, label: '课程排得满，实习要牺牲学分' },
        { v: 0.72, label: '学校不支持 / 地理位置不允许' }
      ]
    },
    {
      // 学长学姐的实际去向，是比任何排名和宣传都可靠的先行指标
      key: 'seniorOutcome', label: '学长学姐的就业情况', weight: 0.24,
      hint: '往届毕业生的去向，比学校官网的就业率真实得多',
      options: [
        { v: 1.26, label: '基本都拿到了满意的 offer' },
        { v: 1.10, label: '大部分顺利就业' },
        { v: 1.00, label: '有人好有人差，正常水平', hint: '基准档' },
        { v: 0.82, label: '普遍焦虑，签得很晚' },
        { v: 0.62, label: '大面积没着落，都在考公考研兜底' }
      ]
    }
  ];

  /* ---------------------------------------------------------------------
   * 主观体验维度
   * ---------------------------------------------------------------------
   * 和客观性价比彻底分开算。一个人完全可能在条件很差的学校过得很开心，
   * 也可能在名校长期抑郁 —— 把这两件事揉进一个数字，两边都会失真。
   * 这组维度不进客观分，只进主观分和总分。
   * ------------------------------------------------------------------- */
  var SUBJECTIVE_DIMS = [
    {
      key: 'likeMajor', label: '喜欢现在这个专业吗', weight: 0.24,
      hint: '四年里你要跟它朝夕相处',
      options: [
        { v: 1.32, label: '很喜欢，愿意一直做下去' },
        { v: 1.14, label: '还行，学得下去' },
        { v: 1.00, label: '无感，就是混个文凭', hint: '基准档' },
        { v: 0.82, label: '不喜欢，学得挺痛苦' },
        { v: 0.62, label: '很排斥，一直想转专业或退学' }
      ]
    },
    {
      key: 'likeSchool', label: '喜欢这所学校吗', weight: 0.20,
      options: [
        { v: 1.28, label: '很喜欢，庆幸自己来了这' },
        { v: 1.12, label: '挺好的，没什么不满' },
        { v: 1.00, label: '说不上喜欢也说不上讨厌', hint: '基准档' },
        { v: 0.84, label: '不太喜欢' },
        { v: 0.66, label: '很后悔来这里' }
      ]
    },
    {
      key: 'pressure', label: '学习压力', weight: 0.19,
      hint: '压力本身不是坏事，扛不住才是',
      options: [
        { v: 1.14, label: '轻松，有大量可自由支配的时间' },
        { v: 1.06, label: '适中，忙但不焦虑' },
        { v: 1.00, label: '有点大，但扛得住', hint: '基准档' },
        { v: 0.84, label: '很大，长期睡不好' },
        { v: 0.64, label: '快撑不住了' }
      ]
    },
    {
      key: 'mood', label: '这一年整体的情绪状态', weight: 0.22,
      hint: '如果长期低落，请优先找学校的心理咨询，而不是看这个分数',
      options: [
        { v: 1.26, label: '大部分时间是开心的' },
        { v: 1.10, label: '还不错' },
        { v: 1.00, label: '平淡，没什么波澜', hint: '基准档' },
        { v: 0.80, label: '经常低落' },
        { v: 0.58, label: '长期情绪很差 / 已经在就医' }
      ]
    },
    {
      key: 'friends', label: '人际关系', weight: 0.15,
      options: [
        { v: 1.22, label: '有几个能交心的朋友' },
        { v: 1.08, label: '有一些玩得来的人' },
        { v: 1.00, label: '泛泛之交', hint: '基准档' },
        { v: 0.86, label: '比较孤独' },
        { v: 0.70, label: '几乎没有朋友' }
      ]
    }
  ];

  /* ---------------------------------------------------------------------
   * 实习
   * ---------------------------------------------------------------------
   * 实习是「大学地理位置」最直接的变现方式，所以它同时进三个地方：
   *   1. 工资 → 从净成本里扣掉（分母变小，得分上升）
   *   2. 含金量与转正概率 → 加成前景系数（分子变大）
   *   3. 工时 + 通勤 → 挤压校园生活（分子变小，这是必须扣的）
   * 只加不扣会让「每天 12 小时实习」看起来是最优解，那显然不对。
   * ------------------------------------------------------------------- */
  var INTERN_QUALITY = [
    { v: 1.00, label: '目前没有实习', hint: '不参与加成' },
    { v: 1.03, label: '校内兼职 / 家教 / 非对口零工', hint: '有现金，但对简历帮助有限' },
    { v: 1.12, label: '一般对口实习', hint: '专业相关，中小公司' },
    { v: 1.22, label: '知名企业对口实习', hint: '简历上写出来别人认得' },
    { v: 1.32, label: '头部企业核心岗实习', hint: '大厂核心业务线 / 顶级机构' }
  ];

  var CONVERT_PROB = [
    { v: 0.00, label: '没有实习 / 不涉及转正' },
    { v: 0.05, label: '基本没有转正名额' },
    { v: 0.20, label: '有机会但不确定（约 20%）' },
    { v: 0.40, label: '表现好就有（约 40%）' },
    { v: 0.65, label: '大概率能转（约 65%）' },
    { v: 1.00, label: '已拿到转正 / return offer' }
  ];

  /* 转正期权加成。含金量与转正概率**不是独立变量** —— 好公司本身就自带
   * 更高更结构化的转正率，直接相乘等于把同一个潜变量的权重平方
   * （头部 1.32 × 转正 1.25 = 1.65，超过两者各自上限之和）。
   * 这里让转正权重随含金量升高而衰减：含金量越高，转正的边际贡献越小。 */
  var CONVERT_MAX_BONUS = 0.25;
  function convertBonusOf(qualityV, probV) {
    var damp = Math.max(2 - qualityV, 0);          // 含金量 1.32 → 阻尼 0.68
    return 1 + probV * CONVERT_MAX_BONUS * damp;
  }
  var PROSPECT_BOOST_CAP = 1.50;                   // 实习对前景的总加成封顶

  /* 时间惩罚曲线（分段）。
   * v1 用「16h 起罚、55h 罚满」的单段线性，导致标准全职实习
   * （40h 工作 + 10h 通勤 ≈ 50h）和 90h 的压榨实习被压进同一个地板，
   * 于是「选全职必然比兼职低分」，与含金量和转正率无关 —— 那是 bug。
   * v2 把「正常全职」和「过劳」分开：
   *   0–20h   不罚（兼职、轻度实习）
   *   20–45h  线性到 −15%（标准全职 + 合理通勤，应当只有温和成本）
   *   45–75h  线性到 −40%（真正的过劳区间）
   *   75h+    封顶 −40%
   */
  var INTERN_TIME = {
    FREE: 20, NORMAL: 45, SEVERE: 75,
    NORMAL_PENALTY: 0.15, MAX_PENALTY: 0.40
  };

  /* 实习收入的抵扣权重：非对口零工只解决现金流，不产生任何职业资本，
   * 却占用了本可用于学习或找对口实习的时间。把它和大厂实习同权抵扣，
   * 会把「穷学生打零工硬扛生活费」误判成顶级性价比。 */
  var INTERN_RELEVANCE = [
    { v: 0.45, label: '完全不对口（便利店 / 服务业 / 派发传单）', hint: '只解决现金流，不积累职业资本' },
    { v: 0.70, label: '部分相关（家教 / 校内助理 / 泛行业）' },
    { v: 1.00, label: '高度对口（就是我未来想做的方向）', hint: '基准档，收入全额抵扣' }
  ];

  /* 学期中实习挤占的是课程时间，假期实习的机会成本接近于零 ——
   * 同样 50 小时/周，两者不该用同一条惩罚曲线。 */
  var INTERN_TERM = [
    { v: 1.00, label: '学期中（要和课程抢时间）' },
    { v: 0.45, label: '寒暑假为主' },
    { v: 0.70, label: '两者都有' }
  ];

  var WEEKS_PER_MONTH = 4.33;

  /* ---------------------------------------------------------------------
   * 评级
   * ------------------------------------------------------------------- */
  /* 阈值按 24 个真实人群样本的实际分布校准过一轮。
   * v1 的阈值下，7/24 落进最高档，「别声张」失去稀缺性；
   * v2 上移后最高档约占 17%，「正常」档仍然锚在 1.00 附近。
   * 注意：样本是刻意挑的有意思的场景，不是随机抽样，
   * 阈值属于「暂定、待真实数据回流后再校准」。 */
  /* 每档都有一个可自称的「名号」（title）+ 一句中性的水平描述（label）。
   * 只给 label 的话，人群众数那一档是「当地正常水平」——
   * 没有人会为「我很正常」发一条朋友圈，中间档等于没有传播力。
   * title 用于分享图和标题，label 保留给需要客观表述的地方。 */
  var RATINGS = [
    { max: 0.55, key: 'bad', title: '渡劫型', label: '性价比很低', emoji: '😵', color: '#e0413a',
      desc: '花的钱和过的日子明显不匹配。先看诊断里最拖后腿的那一项 —— 住宿和地段通常是能动的，专业和学校不能。' },
    { max: 0.78, key: 'low', title: '为爱发电型', label: '偏亏', emoji: '😟', color: '#e8763a',
      desc: '你在为这段大学生活付溢价。如果换宿舍、换住法、或把花销压回当地水平，分数会明显上来。' },
    { max: 0.95, key: 'below', title: '差一口气型', label: '略低于当地水平', emoji: '😐', color: '#d9a13a',
      desc: '不算糟，但同样的钱在当地能换到更好的居住条件。值得花一个周末认真比较一下校外租房。' },
    { max: 1.30, key: 'mid', title: '标准大学生', label: '当地正常水平', emoji: '🙂', color: '#4aa3e0',
      desc: '不亏不赚，把钱花在了该花的地方 —— 这其实是最多人待的位置。想往上走，看看诊断里哪一项拨一档能拉动最多。' },
    { max: 1.75, key: 'good', title: '悄悄划算型', label: '相当划算', emoji: '😀', color: '#2f9e6e',
      desc: '花的钱换来了高于当地平均的居住与发展条件。这个组合值得守住。' },
    { max: 2.60, key: 'great', title: '别人家的大学', label: '非常值', emoji: '🤩', color: '#7b4fd9',
      desc: '住得好、位置好、前景好，同时没有为此多花钱。这三样能同时成立的学校不多。' },
    { max: Infinity, key: 'dream', title: '别声张', label: '天花板', emoji: '👑', color: '#c9971f',
      desc: '要么你的实习已经把生活费全覆盖了还带转正预期，要么条件好得不像话。' +
            '如果你没有实习却拿到这个分，回去核对一下花销那一栏 —— 多半是漏填了。' }
  ];

  var RADAR_DIMS = [
    { key: 'dorm',     label: '住宿' },
    { key: 'location', label: '地段' },
    // 叫「花销」有歧义：82 分到底是花得多还是花得划算？
    // 统一成「高 = 好」的命名，读者不用再猜方向。
    { key: 'cost',     label: '划算' },
    { key: 'prospect', label: '前景' },
    { key: 'campus',   label: '校园' }
  ];

  /* ---------------------------------------------------------------------
   * 表单结构
   * ------------------------------------------------------------------- */
  function opts(arr) {
    return arr.map(function (o, i) { return { value: i, label: o.label, hint: o.hint }; });
  }
  function countryByKey(cc) {
    for (var i = 0; i < COUNTRIES.length; i++) if (COUNTRIES[i].key === cc) return COUNTRIES[i];
    return COUNTRIES[0];
  }

  function buildSections(cc) {
    var C = countryByKey(cc);
    return [
      {
        key: 'basic', title: '学校与地点', icon: '①',
        desc: '学校名和专业只用于显示，不参与计算',
        fields: [
          { key: 'schoolName', type: 'text', label: '学校名称', def: '', placeholder: '例：海南大学 / Universitat Autònoma de Barcelona' },
          { key: 'majorName',  type: 'text', label: '专业名称', def: '', placeholder: '例：软件工程' },
          { key: 'stage', type: 'select', label: '当前阶段', def: 0,
            options: [{ value: 0, label: '本科生' }, { value: 1, label: '授课型硕士' }],
            hint: '本工具不适用于博士与研究型硕士' },
          { key: 'region', type: 'select', label: '城市层级', def: Math.min(2, C.regions.length - 1),
            options: C.regions.map(function (r, i) { return { value: i, label: r.label }; }),
            hint: '用于确定「当地基准花销」，是分母的锚' },
          { key: 'location', type: 'select', label: '校区地段', def: 2,
            options: opts(LOCATIONS), hint: '同时影响生活便利度和当地基准房租' },
          { key: 'commute', type: 'select', label: '进城交通', def: 1, options: opts(COMMUTE) }
        ]
      },
      {
        key: 'cost', title: '花销', icon: '②',
        desc: '按本币填，境外会自动按购买力平价折算成人民币',
        fields: [
          { key: 'monthlyLiving', type: 'number', label: '每月生活费（吃饭 + 日常，不含住宿）',
            unit: C.cur, def: C.regions[Math.min(2, C.regions.length - 1)].baseLiving, min: 0, step: 50,
            hint: '包含三餐、话费、日用品、社交娱乐' },
          { key: 'monthlyHousing', type: 'number', label: '每月住宿花销（宿舍费摊到月 / 房租）',
            unit: C.cur, def: C.regions[Math.min(2, C.regions.length - 1)].baseHousing, min: 0, step: 50,
            hint: '住宿舍的话，用「年住宿费 ÷ 10」填' },
          { key: 'monthlyOther', type: 'number', label: '其他固定月支出（可选）',
            unit: C.cur, def: 0, min: 0, step: 50,
            hint: '健身卡、交通月票、固定的课外班等；学费不填' },
          { key: 'costSource', type: 'select', label: '这笔钱主要来自', def: 0,
            options: [
              { value: 0, label: '家里给' },
              { value: 1, label: '家里给 + 自己兼职补贴' },
              { value: 2, label: '奖学金 / 助学金' },
              { value: 3, label: '完全自己赚' },
              { value: 4, label: '助学贷款' }
            ],
            hint: '不参与计算，仅在分享图上标注' }
        ]
      },
      { key: 'dorm',     title: '住宿条件', icon: '③', desc: '决定你每天 10 小时待在哪儿', fields: dims2fields(DORM_DIMS) },
      { key: 'campus',   title: '校园生活', icon: '④', desc: '门禁、管理、食堂、设施、氛围', fields: dims2fields(CAMPUS_DIMS) },
      { key: 'prospect', title: '专业与前景', icon: '⑤', desc: '这四年之后能换来什么',       fields: dims2fields(PROSPECT_DIMS) },
      { key: 'subj',     title: '主观感受', icon: '⑦',
        desc: '这一组只进「主观体验分」，不参与客观性价比的计算',
        fields: dims2fields(SUBJECTIVE_DIMS) },
      {
        key: 'intern', title: '实习', icon: '⑥',
        desc: '学校的地理位置能不能变现，主要看这一栏。工资抵成本、转正加前景、工时扣生活',
        fields: [
          { key: 'internQuality', type: 'select', label: '实习含金量', def: 0, options: opts(INTERN_QUALITY) },
          { key: 'internRelevance', type: 'select', label: '与专业的对口程度', def: 2, options: opts(INTERN_RELEVANCE),
            hint: '不对口的零工只解决现金流，抵扣权重会打折' },
          { key: 'internTerm', type: 'select', label: '主要发生在', def: 0, options: opts(INTERN_TERM),
            hint: '假期实习几乎不占用课程时间，时间惩罚会大幅减轻' },
          { key: 'internDaysPerWeek', type: 'number', label: '每周实习天数', unit: '天', def: 0, min: 0, max: 7, step: 1 },
          { key: 'internHours', type: 'number', label: '每天工作时长', unit: '小时', def: 8, min: 0, max: 16, step: 0.5 },
          { key: 'internCommute', type: 'number', label: '每天通勤合计', unit: '小时', def: 1, min: 0, max: 8, step: 0.5,
            hint: '往返加起来。远郊校区的通勤成本往往就吃在这里' },
          { key: 'internDailyPay', type: 'number', label: '实习日薪', unit: C.cur, def: 0, min: 0, step: 10,
            hint: '按日结算填日薪；按月发的话用「月薪 ÷ 当月实习天数」' },
          { key: 'convertProb', type: 'select', label: '转正概率', def: 0, options: opts(CONVERT_PROB),
            hint: '靠这所学校拿到能转正的实习，是它最大的隐性回报' }
        ]
      }
    ];
  }

  function dims2fields(dims) {
    return dims.map(function (d) {
      var def = 0;
      for (var i = 0; i < d.options.length; i++) if (d.options[i].v === 1.00) { def = i; break; }
      return { key: d.key, type: 'select', label: d.label, def: def,
               options: opts(d.options), hint: d.hint, weight: d.weight };
    });
  }

  root.UNI_DATA = {
    GLOBAL: GLOBAL,
    COUNTRIES: COUNTRIES,
    countryByKey: countryByKey,
    LOCATIONS: LOCATIONS,
    COMMUTE: COMMUTE,
    DORM_DIMS: DORM_DIMS,
    CAMPUS_DIMS: CAMPUS_DIMS,
    PROSPECT_DIMS: PROSPECT_DIMS,
    SUBJECTIVE_DIMS: SUBJECTIVE_DIMS,
    SUBJ_EXP: 1.35,          // 主观分放大指数：让 0.6~1.3 的加权均值撑开到可读区间
    /* 总分 = 客观 × 主观^SUBJ_WEIGHT。
     * 不用「客观^0.6 × 主观^0.4」那种写法 —— 那样即使主观是中性的 1.00，
     * 总分也会变成 客观^0.6，把所有分数往 1 压（实测范围从 0.21~3.80
     * 缩到 0.39~2.22），已校准的阈值全部失效。写成乘数形式，
     * 主观中性时总分恰好等于客观分，阈值可以继续沿用。 */
    SUBJ_WEIGHT: 0.45,
    INTERN_QUALITY: INTERN_QUALITY,
    INTERN_RELEVANCE: INTERN_RELEVANCE,
    INTERN_TERM: INTERN_TERM,
    CONVERT_PROB: CONVERT_PROB,
    CONVERT_MAX_BONUS: CONVERT_MAX_BONUS,
    convertBonusOf: convertBonusOf,
    PROSPECT_BOOST_CAP: PROSPECT_BOOST_CAP,
    INTERN_TIME: INTERN_TIME,
    HOUSING_BASE: HOUSING_BASE,
    WEEKS_PER_MONTH: WEEKS_PER_MONTH,
    RATINGS: RATINGS,
    RADAR_DIMS: RADAR_DIMS,
    buildSections: buildSections
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = root.UNI_DATA;

})(typeof window !== 'undefined' ? window : globalThis);
