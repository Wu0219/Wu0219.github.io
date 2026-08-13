/* ===========================================================================
 * data.js —— 程序员版工作性价比计算器 · 国际版系数表
 * ---------------------------------------------------------------------------
 * 结构：
 *   GLOBAL      —— 跨国通用常数
 *   通用系数表   —— 技术成长 / 团队环境 / 股票折价 / 通勤 / On-call …（全球一致）
 *   COUNTRIES   —— 按国家分表：薪资曲线、城市、公司类型、税制、社保、学历、年龄风险
 *   buildSections(countryKey) —— 按国家生成表单
 *
 * 想改模型？只改这一个文件。改完跑 `node assets/tests.js` 看刻度有没有调歪。
 * =========================================================================== */

(function (root) {
  'use strict';

  /* =========================================================================
   * 一、跨国通用常数
   * ========================================================================= */
  var GLOBAL = {
    WEEKS_PER_YEAR: 52,
    SICK_LEAVE_WEIGHT: 0.6,   // 带薪病假只按 0.6 天折算
    SLACK_WEIGHT: 0.5,        // 摸鱼按半价折回
    MIN_EFF_HOURS: 4,         // 有效工时下限
    STD_DAY_HOURS: 8,         // 超过这个才算加班

    // PPP 锚：所有国家的钱都折算到「中国等效人民币购买力」，用于跨国对比。
    // 数值为各国 PPP 换算因子（本币 / 国际元），来源同 zippland/worth-calculator。
    PPP_ANCHOR: 4.19
  };

  /* -------------------------------------------------------------
   * 股票 / 期权折价系数  [分子·作用于股票面值]  —— 全球通用
   * ----------------------------------------------------------- */
  var STOCK_DISCOUNT = [
    { v: 0,    label: '没有股票/期权',                hint: '总包就是现金' },
    { v: 0.90, label: '已上市 RSU · 可自由变现',       hint: '归属后能卖，打 9 折算流动性与股价波动' },
    { v: 0.72, label: '已上市 RSU · 股价长期阴跌',     hint: '发的时候好看，归属时缩水' },
    { v: 0.35, label: '未上市期权 · 有回购或上市预期',  hint: '大概率能兑现一部分，但周期长、可能打折回购' },
    { v: 0.15, label: '早期期权 · B 轮 / Serie B 以前', hint: '统计上绝大多数归零，按彩票期望值算' },
    { v: 0.03, label: '口头承诺 / 明确画饼',            hint: '没有书面协议的一律按接近 0 处理' }
  ];

  var OVERTIME_COMP = [
    { v: 1.00, label: '无任何补偿',              hint: '白干，加班时间 100% 计入成本' },
    { v: 0.92, label: '有调休但基本休不掉',       hint: '名义上有，实际攒着过期' },
    { v: 0.85, label: '调休能真正休掉',           hint: '时间换时间' },
    { v: 0.75, label: '1.5 倍加班费',            hint: '按劳动法工作日标准' },
    { v: 0.65, label: '2 倍及以上 / 加班补贴',    hint: '周末双倍或有明确的加班补贴' }
  ];

  var COMMUTE_COMFORT = [
    { v: 0.60, label: '步行 / 骑车 15 分钟内',   hint: '几乎无损耗，还能当运动' },
    { v: 0.75, label: '有座地铁 / 公司班车',     hint: '能看书能补觉' },
    { v: 1.00, label: '普通地铁公交',            hint: '基准' },
    { v: 1.15, label: '换乘多次 / 极度拥挤',     hint: '到工位已耗掉一半电' },
    { v: 1.05, label: '自驾且经常堵车',          hint: '有座但精神紧张，还有油费停车费' }
  ];

  var ONCALL = [
    { v: 0.0, label: '完全没有值班',                  hint: '下班即失联' },
    { v: 0.3, label: '轻度 · 每月轮几天，很少被叫',    hint: '有心理负担但很少真被打扰' },
    { v: 0.8, label: '中度 · 每月一周，偶尔半夜起',    hint: '轮到那周不敢喝酒不敢出远门' },
    { v: 1.5, label: '重度 · 长期待命，常半夜处理',    hint: '手机永远静音不了' },
    { v: 2.5, label: '地狱 · 随时电话，节假日无休',     hint: '本质是 24 小时待命岗' }
  ];

  var LEAVE_HARD = [
    { v: 1.00, label: '说走就走，不用解释',    hint: '年假全额有效' },
    { v: 0.85, label: '要审批，基本都能批',    hint: '轻微折损' },
    { v: 0.50, label: '很难请，经常被劝退',    hint: '一半的年假是纸面数字' },
    { v: 0.15, label: '形同虚设 / 从没休过',   hint: '写在合同里的装饰品' }
  ];

  /* -------------------------------------------------------------
   * 技术成长维度  [分子·加权平均]  —— 全球通用
   * ----------------------------------------------------------- */
  var GROWTH_DIMS = [
    {
      key: 'techStack', label: '技术栈市场价值', weight: 0.24,
      hint: '决定你换工作时简历上那几行值多少钱',
      options: [
        { v: 1.30, label: 'AI / 大模型 / 算法',            hint: 'LLM、推荐、CV/NLP、AI Infra —— 2025 全球薪资增长最快的方向' },
        { v: 1.22, label: '基础架构 / 云原生 / 内核',       hint: 'K8s、数据库内核、编译器、中间件，稀缺且抗贬值' },
        { v: 1.08, label: '音视频 / 图形 / 游戏引擎',       hint: '门槛高，圈子小但需求稳定' },
        { v: 1.05, label: '后端 / 微服务（Java·Go·Rust）',  hint: '盘子最大的主流方向' },
        { v: 1.00, label: '前端 / 客户端（Web·iOS·安卓）',  hint: '基准档' },
        { v: 1.00, label: '数据开发 / 数仓 / BI',           hint: '基准档，向 AI 方向迁移性好' },
        { v: 0.98, label: '运维 / SRE / DevOps',            hint: '稳定但天花板受限' },
        { v: 0.98, label: '嵌入式 / 驱动 / 固件',           hint: '换工作面窄，但不容易被裁' },
        { v: 0.90, label: '测试开发 / QA',                  hint: '市场需求与议价能力偏弱' },
        { v: 0.82, label: '传统企业开发（ERP/SAP/政企）',    hint: '技术更新慢，跳槽时溢价低' },
        { v: 0.78, label: '低代码 / 配置化平台',            hint: '主要在拖拽和填配置，代码能力会退化' },
        { v: 0.65, label: '祖传技术栈（COBOL/Delphi/VB/PB）', hint: '离开这家公司几乎无处可去' }
      ]
    },
    {
      key: 'autonomy', label: '技术自主度', weight: 0.20,
      hint: 'Stack Overflow 2025 调查里，「自主权与信任」是程序员满意度的第 1 位因素，高于薪酬',
      options: [
        { v: 1.25, label: '核心自研 · 我能定技术选型',  hint: '从 0 到 1，方案我说了算' },
        { v: 1.10, label: '参与核心系统开发',           hint: '写真正跑在核心链路上的代码' },
        { v: 1.00, label: '二次开发 / 调接口拼装',      hint: '基准档，大部分业务开发的日常' },
        { v: 0.85, label: '改配置 / 写业务胶水代码',    hint: '技术含量低，可替代性高' },
        { v: 0.70, label: '外派驻场 / 按甲方文档搬砖',  hint: '不参与设计，只交付工时' }
      ]
    },
    {
      key: 'codebase', label: '代码库质量（屎山程度）', weight: 0.15,
      hint: '你每天要在什么样的地基上盖楼',
      options: [
        { v: 1.15, label: '新项目 · 干净整洁',        hint: '技术债为零，改起来爽' },
        { v: 1.00, label: '有人持续维护 · 还算清晰',   hint: '基准档' },
        { v: 0.90, label: '有点乱 · 但能看懂',        hint: '偶尔踩坑' },
        { v: 0.78, label: '屎山 · 改一处崩三处',      hint: '大部分时间在还债不是在建设' },
        { v: 0.65, label: '考古现场 · 原作者已离职',   hint: '没有文档没有测试没人敢动' }
      ]
    },
    {
      key: 'engineering', label: '工程规范', weight: 0.14,
      hint: '有没有把你训练成一个「专业」的工程师',
      options: [
        { v: 1.20, label: 'CR + 单测 + CI/CD + 文档齐全', hint: '在这里待两年，工程习惯就成型了' },
        { v: 1.10, label: '有 Code Review 和 CI',        hint: '基本工程素养有保障' },
        { v: 1.00, label: '有 Git，其余看个人自觉',       hint: '基准档' },
        { v: 0.85, label: '全靠手动，上线靠 SSH',         hint: '学不到现代工程实践' },
        { v: 0.70, label: '没版本管理 / 直接改生产',      hint: '这不是技术团队，是事故现场' }
      ]
    },
    {
      key: 'bizProspect', label: '业务线前景', weight: 0.12,
      hint: '你在的这条船是在往上走还是在沉',
      options: [
        { v: 1.20, label: '高增长 · 公司核心业务',   hint: '资源倾斜，涨薪晋升都优先' },
        { v: 1.05, label: '稳定盈利 · 现金牛',       hint: '不刺激但安稳' },
        { v: 1.00, label: '平稳 · 不好不坏',         hint: '基准档' },
        { v: 0.85, label: '边缘业务 · 老板不太关心',  hint: '晋升难，容易被优化' },
        { v: 0.70, label: '随时可能砍掉',            hint: '每次开会都在讨论要不要继续做' }
      ]
    },
    {
      key: 'codingRatio', label: '实际写代码时间占比', weight: 0.09,
      hint: '会议、对齐、写文档也算工作，但不长技术',
      options: [
        { v: 1.10, label: '60% 以上 · 大部分时间在写码', hint: '专注度高' },
        { v: 1.00, label: '40% ~ 60%',                  hint: '基准档' },
        { v: 0.90, label: '20% ~ 40% · 会议很多',        hint: '一天被切成碎片' },
        { v: 0.80, label: '20% 以下 · 全在开会写文档',    hint: '本质已经不是开发岗了' }
      ]
    },
    {
      key: 'learning', label: '学习与技术分享', weight: 0.06,
      hint: '公司愿不愿意为你的成长花时间',
      options: [
        { v: 1.15, label: '有技术分享文化 / 允许投入时间', hint: '内部分享、技术评审、鼓励开源' },
        { v: 1.00, label: '偶尔有，看个人',              hint: '基准档' },
        { v: 0.90, label: '完全没有，只看交付',           hint: '想学只能靠下班后的自己' }
      ]
    }
  ];

  /* -------------------------------------------------------------
   * 团队与人文环境  [分子·加权平均]  —— 全球通用
   * ----------------------------------------------------------- */
  var ENV_DIMS = [
    {
      key: 'leader', label: '直属领导', weight: 0.28,
      hint: '程序员离职原因排行第一，从来都不是钱',
      options: [
        { v: 1.25, label: '技术出身 · 扛事 · 护犊子', hint: '能给你挡需求、争资源、背锅' },
        { v: 1.10, label: '懂技术 · 讲道理',          hint: '至少不会让你三天做完三周的活' },
        { v: 1.00, label: '中规中矩 · 各司其职',      hint: '基准档' },
        { v: 0.85, label: '外行指挥内行',             hint: '技术方案要靠你说服一个不懂的人' },
        { v: 0.70, label: 'PUA 型 / 甩锅型',          hint: '出事你背，出功他领' }
      ]
    },
    {
      key: 'team', label: '团队技术水平', weight: 0.24,
      hint: '同事的水平决定你的成长速度上限',
      options: [
        { v: 1.20, label: '大牛云集 · 能学到东西',  hint: 'CR 里能被指出真问题' },
        { v: 1.10, label: '有几个靠谱的能请教',      hint: '遇到坑有人捞' },
        { v: 1.00, label: '半斤八两 · 各写各的',    hint: '基准档' },
        { v: 0.85, label: '我在带整个团队',         hint: '你的时间全花在给别人擦屁股' },
        { v: 0.75, label: '全是坑 · 天天替人填',    hint: '负成长环境' }
      ]
    },
    {
      key: 'requirement', label: '需求质量 / 产品靠谱度', weight: 0.24,
      hint: '需求变更是程序员加班的头号来源',
      options: [
        { v: 1.15, label: '需求清晰 · 有评审 · 少变更', hint: '排期是可信的' },
        { v: 1.00, label: '一般 · 偶尔改改',            hint: '基准档' },
        { v: 0.85, label: '经常变 · 做完再推翻',        hint: '重复劳动多' },
        { v: 0.70, label: '老板拍脑袋 · 需求日更',      hint: '上线前一晚还在改主流程' }
      ]
    },
    {
      key: 'device', label: '开发设备', weight: 0.13,
      hint: '每天 8 小时面对的生产工具',
      options: [
        { v: 1.10, label: '顶配 Mac / 高配机 + 双 4K', hint: '编译不用等，多窗口不卡' },
        { v: 1.00, label: '够用 · 不影响开发',         hint: '基准档' },
        { v: 0.90, label: '老旧卡顿 · 编译要等很久',    hint: '每天被机器偷走一小时' },
        { v: 0.80, label: '自带设备 / 远程虚拟机卡',    hint: '公司连生产工具都不给' }
      ]
    },
    {
      key: 'office', label: '办公环境', weight: 0.11,
      hint: '心流被打断的成本很高',
      options: [
        { v: 1.05, label: '安静 · 独立工位 / 可戴耳机', hint: '能进入心流' },
        { v: 1.00, label: '普通开放式办公',            hint: '基准档' },
        { v: 0.95, label: '嘈杂 / 工位很挤',           hint: '经常被打断' },
        { v: 0.90, label: '无固定工位 / 客户现场',      hint: '连桌子都不属于你' }
      ]
    }
  ];

  var PERKS = [
    { key: 'canteen',   v: 0.030, label: '食堂 / 餐补（ticket restaurant）' },
    { key: 'shuttle',   v: 0.020, label: '通勤班车 / 交通补贴' },
    { key: 'gym',       v: 0.015, label: '健身房 / 运动补贴' },
    { key: 'medical',   v: 0.025, label: '补充商业医疗保险' },
    { key: 'housing',   v: 0.030, label: '住房补贴 / 人才公寓' },
    { key: 'equipment', v: 0.015, label: '设备可自选 / 有软件采购预算' },
    { key: 'flexible',  v: 0.030, label: '弹性工时 / 不打卡' },
    { key: 'training',  v: 0.015, label: '培训 / 技术大会 / 认证报销' }
  ];

  /* -------------------------------------------------------------
   * 风险加点  [分母·相加]  —— 全球通用
   * ----------------------------------------------------------- */
  var RISK_DIMS = [
    {
      key: 'bizHealth', label: '业务线健康度',
      options: [
        { v: 0.00, label: '公司核心业务 · 资源充足' },
        { v: 0.02, label: '稳定盈利' },
        { v: 0.10, label: '边缘业务 · 存在感低' },
        { v: 0.22, label: '亏损 / 传闻要砍' }
      ]
    },
    {
      key: 'layoff', label: '裁员情况',
      options: [
        { v: 0.00, label: '没听说过裁员' },
        { v: 0.08, label: '有传闻 · 人心浮动' },
        { v: 0.15, label: '近一年裁过一轮' },
        { v: 0.25, label: '正在裁 / 已收到风声' }
      ]
    },
    {
      key: 'salaryInvert', label: '薪资倒挂',
      options: [
        { v: 0.00, label: '没有倒挂' },
        { v: 0.05, label: '轻微 · 新人和我差不多' },
        { v: 0.12, label: '严重 · 新人明显比我高' }
      ]
    },
    {
      key: 'payDelay', label: '发薪与社保',
      options: [
        { v: 0.00, label: '按时发 · 社保足额' },
        { v: 0.06, label: '社保按最低基数缴' },
        { v: 0.14, label: '偶尔延迟发薪' },
        { v: 0.32, label: '拖欠工资 / 不交社保' }
      ]
    }
  ];

  /* =========================================================================
   * 二、按国家分表
   * ========================================================================= */

  /* ---------- 中国大陆 ---------- */
  var CN = {
    key: 'CN', name: '中国大陆', flag: '🇨🇳', cur: '¥', curName: '元',
    ppp: 4.19,

    // 基准时薪（本币/小时）：一个「应届 / 双非本科 / 中位城市 / 中位环境」的程序员，
    // 每付出一小时有效工时，市场应该给的钱。
    // 标定：应届中位年包 ≈ 15.6 万 ÷ 年有效工作日 236 ÷ 日均有效工时 11 ≈ 60
    baseHourly: 60,

    /* 税后锚修正系数 = 中位程序员的 (税后 TC ÷ 税前 TC)
     * baseHourly 是按【税前】中位数标定的。用户打开「按税后口径计算」时，
     * 分子换成了税后 TC，分母的锚却还是税前口径 —— 会把所有人系统性压分。
     * 打开税后时，基准时薪按此系数同步下调，保证刻度一致。
     * 中国：月 11,143 × 14 薪 + 8% 公积金 → 税前 16.67 万 / 税后 14.99 万 = 0.899 */
    taxRatio: 0.899,

    salaryMode: 'monthly',
    hasHousingFund: true,

    labels: {
      salaryTitle: '薪酬包（TC）',
      salaryDesc: '程序员的总包不是月薪 ×12。把年终、股票、公积金、补贴全部折进来，才是真实收入。',
      leaveHint: '国内常见 5~15 天；法定假日 13 天',
      taxName: '个人所得税'
    },

    /* 工作年限 → 基准薪资倍数（相对应届）
     * 标定：应届锚 ≈ 15.6 万；×2.25 ≈ 35 万（大厂 P6 / 字节 2-1 的下沿）；
     *       ×2.95 ≈ 46 万（5~8 年）；×3.45 ≈ 54 万（P7 起步区间）。
     * 曲线整体略低于大厂实际给薪 —— 大厂只是市场头部，
     * 拿头部当「应得」标准会让绝大多数人无脑不及格。 */
    years: [
      { v: 1.00, label: '应届 / 1 年以内' },
      { v: 1.55, label: '1 ~ 3 年' },
      { v: 2.25, label: '3 ~ 5 年' },
      { v: 2.95, label: '5 ~ 8 年' },
      { v: 3.45, label: '8 ~ 10 年' },
      { v: 3.75, label: '10 ~ 15 年' },
      { v: 3.90, label: '15 年以上' }
    ],

    cities: [
      { v: 0.72, label: '北京 / 上海 / 深圳', hint: '房租与生活成本天花板' },
      { v: 0.80, label: '广州 / 杭州',        hint: '一线水平，成本略低' },
      { v: 0.88, label: '新一线（成都·武汉·南京·苏州·西安·长沙…）', hint: '性价比常见甜点区' },
      { v: 1.00, label: '二线城市',           hint: '基准档' },
      { v: 1.10, label: '三线城市',           hint: '同样的钱明显更经花' },
      { v: 1.22, label: '四线 / 县城',        hint: '生活成本很低，但机会也少' }
    ],

    companyTypes: [
      { key: 'bigtech',   label: '一线大厂（BAT·字节·美团·华为·网易…）', track: 1.00, risk: 1.00, market: 1.0,
        hint: '涨薪最快，所以对薪资的期望也最高；稳定性中等' },
      { key: 'listed',    label: '二线大厂 / 已上市中厂',   track: 0.92, risk: 1.02, market: 1.0,
        hint: '比一线略缓，风险相当' },
      { key: 'unicorn',   label: '独角兽 / 明星创业（D 轮后）', track: 0.95, risk: 1.12, market: 1.0,
        hint: '给得起钱，但业务与融资波动大' },
      { key: 'startup',   label: '早期创业（A/B 轮及以前）', track: 0.90, risk: 1.28, market: 1.0,
        hint: '高风险高波动，期权大概率归零' },
      { key: 'foreign',   label: '外企 / 跨国研发中心',     track: 0.85, risk: 0.95, market: 0.6,
        hint: '涨薪慢但守法，年龄歧视相对轻' },
      { key: 'sme',       label: '中小私企',               track: 0.75, risk: 1.10, market: 1.0,
        hint: '涨薪空间有限，抗风险能力弱' },
      { key: 'soe',       label: '国企 / 银行 / 运营商',    track: 0.45, risk: 0.88, market: 0.25,
        hint: '钱少事稳，对薪资期望大幅下调' },
      { key: 'gov',       label: '体制内 / 高校 / 研究所',  track: 0.32, risk: 0.82, market: 0.2,
        hint: '几乎不涨薪，但也几乎不裁员' },
      { key: 'outsource', label: '外包 / 人力外派 / 驻场',  track: 0.58, risk: 1.25, market: 1.1,
        hint: '涨薪慢、简历贬值、随时被换人' },
      { key: 'freelance', label: '自由职业 / 独立开发',     track: 1.00, risk: 1.18, market: 0.8,
        hint: '完全市场化，收入不稳定但天花板自己定' }
    ],

    contractDim: {
      key: 'contract', label: '用工形式',
      options: [
        { v: 0.00, label: '正式劳动合同' },
        { v: 0.10, label: '实习 / 兼职 / 非全日制' },
        { v: 0.15, label: '劳务派遣（第三方签合同）' },
        { v: 0.08, label: '试用期未过' }
      ]
    },

    /* 年龄风险：互联网约 32 岁出现薪资拐点；某大厂 35 岁以上研发占比不足 7%；
     * 超 60% 岗位明确要求 35 岁以下。再乘公司类型的 market 调节。 */
    ageRisk: [
      { max: 28,       v: 0.00, label: '28 岁以下' },
      { max: 32,       v: 0.03, label: '28 ~ 32 岁' },
      { max: 35,       v: 0.08, label: '32 ~ 35 岁' },
      { max: 40,       v: 0.15, label: '35 ~ 40 岁' },
      { max: 45,       v: 0.20, label: '40 ~ 45 岁' },
      { max: Infinity, v: 0.24, label: '45 岁以上' }
    ],

    degrees: [
      { key: 'below',    label: '专科及以下' },
      { key: 'bachelor', label: '本科' },
      { key: 'master',   label: '硕士' },
      { key: 'phd',      label: '博士' }
    ],
    schools: [
      { key: 'tier3', label: '二本 / 三本 / 民办' },
      { key: 'tier2', label: '双非一本 / QS200 / USnews80' },
      { key: 'tier1', label: '985 211 / QS50 / USnews30' }
    ],
    // 程序员行业的学历溢价明显小于通用行业，跨度已收窄
    eduTable: {
      below:       { fixed: 0.90 },
      bachelor:    { tier3: 0.97, tier2: 1.00, tier1: 1.08 },
      masterBase:  { tier3: 0.97, tier2: 1.00, tier1: 1.08 },
      masterBonus: { tier3: 0.12, tier2: 0.15, tier1: 0.18 },
      phd:         { tier3: 1.30, tier2: 1.38, tier1: 1.45 }
    },

    // 个人社保（养老 8% + 医疗 2% + 失业 0.5%），公积金另算
    social: { rate: 0.105, capMonthly: 36000, fundRates: [0.05, 0.07, 0.08, 0.10, 0.12] },

    // 2025 年度综合所得税率表（超额累进 + 速算扣除数）
    tax: {
      type: 'quick',
      basicDeduction: 60000,
      brackets: [
        [36000,    0.03, 0],
        [144000,   0.10, 2520],
        [300000,   0.20, 16920],
        [420000,   0.25, 31920],
        [660000,   0.30, 52920],
        [960000,   0.35, 85920],
        [Infinity, 0.45, 181920]
      ]
    },

    defaults: {
      monthlyBase: 25000, salaryMonths: 14, bonusCash: 0, allowanceMonthly: 0,
      stockAnnual: 0, stockType: 0, fundRate: 2, includeCompanyFund: true,
      afterTax: false, specialDeduct: 1500, socialCap: 36000,
      workDaysPerWeek: 2, dailyHours: 10, commuteHours: 1.5, commuteComfort: 2,
      slackHours: 1.5, annualLeave: 10, leaveHard: 1, publicHolidays: 13, sickLeave: 3,
      city: 2, companyType: 1, age: 29, degree: 1, school: 1, bachelorSchool: 1, workYears: 2
    },

    weekDayOptions: [
      { v: 4,   label: '4 天（四天工作制）' },
      { v: 4.5, label: '4.5 天' },
      { v: 5,   label: '5 天（双休）' },
      { v: 5.5, label: '5.5 天（大小周）' },
      { v: 6,   label: '6 天（单休）' },
      { v: 6.5, label: '6.5 天' },
      { v: 7,   label: '7 天（无休）' }
    ],

    sources: [
      '基准时薪 ¥60：2025 全国程序员应届中位年包约 15~16 万 ÷ 年有效工作日 236 天 ÷ 日均有效工时 11 小时',
      '年限倍数与 2025 大厂职级薪资交叉校验（阿里 P5 41-48万 / P6 48-64万 / P7 91-118万；字节 2-1 53-71万；美团 L7 54-71万；华为 15 级 40-47万）',
      '工时：2025 中国程序员平均周工时 48.3 小时，仅约 50% 能拿到加班费或调休',
      '年龄风险：约 32 岁薪资拐点；某大厂 35 岁以上研发占比 <7%；超 60% 岗位限 35 岁以下',
      '个税：2025 年度综合所得 7 级超额累进；社保个人部分按养老 8% + 医疗 2% + 失业 0.5%'
    ]
  };

  /* ---------- 西班牙 ---------- */
  var ES = {
    key: 'ES', name: '西班牙 · España', flag: '🇪🇸', cur: '€', curName: '欧元',
    ppp: 0.62,

    // 基准时薪（€/小时）：一个「应届 / Grado / 中位城市（塞维利亚·萨拉戈萨档）」的程序员。
    // 标定：非马德里/巴塞的 junior 中位 ≈ €22,000 ÷ 年有效工作日 224 ÷ 日均有效工时 8.25 ≈ 11.9
    baseHourly: 12.0,

    /* 税后锚修正系数，含义同中国。
     * 西班牙：bruto €22,000 → SS €1,426 + IRPF €2,781 → neto €17,794 = 0.809
     * 注意这个数明显低于中国的 0.899 —— 如果不做这个修正，
     * 打开税后后西班牙会比中国多被压 11%，跨国比较直接失真。 */
    taxRatio: 0.809,

    salaryMode: 'annual',
    hasHousingFund: false,

    labels: {
      salaryTitle: '薪酬包（Salario bruto）',
      salaryDesc: '西班牙按 salario bruto anual 谈薪，12 或 14 pagas 不影响年总额。把 variable、股票、餐补折进来才是真实收入。',
      leaveHint: '法定最低 22 días laborables，convenio 常给 23~25；festivos 14 天（12 全国+自治区 + 2 地方）',
      taxName: 'IRPF'
    },

    /* 工作年限 → 基准薪资倍数（相对 junior）
     * 标定（bruto anual，全国口径）：
     *   junior 22~26k → 1.00 ；2-3 年 30~34k → 1.28 ；3-5 年 38~45k → 1.68
     *   5-8 年 48~55k → 2.08 ；8-10 年 55~62k → 2.40 ；10-15 年 60~70k → 2.64
     * 曲线比中国平得多 —— 西班牙的涨薪天花板明显更低，这是两国最大的结构差异。 */
    years: [
      { v: 1.00, label: 'Junior / 1 年以内' },
      { v: 1.28, label: '1 ~ 3 年' },
      { v: 1.68, label: '3 ~ 5 年' },
      { v: 2.08, label: '5 ~ 8 年' },
      { v: 2.40, label: '8 ~ 10 年' },
      { v: 2.64, label: '10 ~ 15 年' },
      { v: 2.80, label: '15 年以上' }
    ],

    // 西班牙国内的生活成本差距远小于中国（PPP 已经处理了国家层面的差异）
    cities: [
      { v: 0.90, label: 'Madrid',                              hint: '薪资最高，但房租吃掉大半' },
      { v: 0.88, label: 'Barcelona',                           hint: '房租全国最贵，薪资略低于马德里' },
      { v: 0.93, label: 'Bilbao / San Sebastián / Palma',      hint: '北部与海岛，成本偏高' },
      { v: 0.98, label: 'Valencia / Málaga',                   hint: '近年涨得快，但仍比马德里便宜' },
      { v: 1.02, label: 'Sevilla / Zaragoza / Alicante',       hint: '基准档' },
      { v: 1.06, label: '中型城市（Valladolid·Murcia·Vigo·Granada…）', hint: '同样的钱更经花' },
      { v: 1.12, label: '小城 / 乡镇（pueblo）',                hint: '成本最低，但本地机会很少' },
      { v: 1.10, label: '100% 远程 · 住在低成本地区',           hint: '拿大城市的薪水，付小地方的房租' }
    ],

    companyTypes: [
      { key: 'bigtech',   label: 'Big Tech / 跨国科技公司（FAANG·Microsoft·Booking…）', track: 1.10, risk: 0.95, market: 0.8,
        hint: '西班牙薪资天花板，通常有 RSU；期望也最高' },
      { key: 'product',   label: '产品公司 / 自研 SaaS',        track: 1.00, risk: 1.00, market: 0.9,
        hint: '基准档。代码归属真实，技术积累最扎实' },
      { key: 'scaleup',   label: 'Scale-up / 独角兽（Serie C+）', track: 1.00, risk: 1.10, market: 0.9,
        hint: '给得起钱，但融资与业务波动大' },
      { key: 'startup',   label: 'Startup（Serie A/B 及以前）',  track: 0.95, risk: 1.25, market: 0.9,
        hint: '高风险，期权大概率归零' },
      { key: 'consultora', label: '大型咨询（Accenture·Indra·NTT·Capgemini…）', track: 0.62, risk: 1.05, market: 0.9,
        hint: '按工时计费，薪资天花板被 convenio 压住；前两年接触面广' },
      { key: 'carnica',   label: 'Cárnica / 人力外派（subcontrata）', track: 0.50, risk: 1.28, market: 1.0,
        hint: '西班牙版「外包驻场」：senior 在马德里也常只有 38~42k，简历贬值快' },
      { key: 'banca',     label: '银行 / 保险 / 传统大企业 IT',  track: 0.72, risk: 0.92, market: 0.5,
        hint: '稳定、955、convenio 保护好，但技术栈老' },
      { key: 'publica',   label: '公共部门 / 大学 / 研究机构',   track: 0.35, risk: 0.78, market: 0.15,
        hint: 'Funcionario 几乎不可能被裁，但涨薪基本固定' },
      { key: 'pyme',      label: 'PYME / 小公司',              track: 0.70, risk: 1.12, market: 0.9,
        hint: '涨薪空间有限，抗风险能力弱' },
      { key: 'freelance', label: 'Autónomo / 自由职业',        track: 1.05, risk: 1.20, market: 0.8,
        hint: '收入不稳定，且 cuota de autónomos 是实打实的成本' }
    ],

    contractDim: {
      key: 'contract', label: '合同类型（tipo de contrato）',
      options: [
        { v: 0.00, label: 'Indefinido（无固定期限）' },
        { v: 0.06, label: 'período de prueba 未过' },
        { v: 0.12, label: 'Temporal / por obra（定期合同）' },
        { v: 0.10, label: 'Prácticas / becario（实习）' },
        { v: 0.14, label: 'Falso autónomo（假自雇）' }
      ]
    },

    /* 年龄风险：西班牙劳动保护强、没有中国式的 35 岁门槛，
     * 但 45 岁以上再就业确实变慢。整体强度只有中国的 1/3 左右。 */
    ageRisk: [
      { max: 35,       v: 0.00, label: '35 岁以下' },
      { max: 45,       v: 0.02, label: '35 ~ 45 岁' },
      { max: 55,       v: 0.05, label: '45 ~ 55 岁' },
      { max: Infinity, v: 0.09, label: '55 岁以上' }
    ],

    degrees: [
      { key: 'below',    label: 'FP / Ciclo Superior（DAM·DAW·ASIR）' },
      { key: 'bachelor', label: 'Grado / Ingeniería' },
      { key: 'master',   label: 'Máster' },
      { key: 'phd',      label: 'Doctorado' }
    ],
    schools: [
      { key: 'tier3', label: '私立 / 知名度较低的大学' },
      { key: 'tier2', label: '普通公立大学' },
      { key: 'tier1', label: 'UPM · UPC · UC3M · UAM · UB 等 / 海外名校' }
    ],
    // 西班牙的学历溢价比中国还小 —— 市场更看 portfolio 和经验
    eduTable: {
      below:       { fixed: 0.93 },
      bachelor:    { tier3: 0.98, tier2: 1.00, tier1: 1.05 },
      masterBase:  { tier3: 0.98, tier2: 1.00, tier1: 1.05 },
      masterBonus: { tier3: 0.04, tier2: 0.06, tier1: 0.09 },
      phd:         { tier3: 1.12, tier2: 1.16, tier1: 1.20 }
    },

    /* Seguridad Social 劳工方 2025：
     *   contingencias comunes 4.70% + desempleo 1.55% + formación 0.10% + MEI 0.13% = 6.48%
     *   base máxima 4,909.50 €/月 → 58,914 €/年 */
    social: { rate: 0.0648, capMonthly: 4909.5, fundRates: null },

    /* IRPF 2025 —— 国家 + 自治区合并后的一般税率阶梯（近似全国平均，
     * 马德里略低、加泰/瓦伦西亚略高）。
     * 计算方式：base = bruto − SS − gastos deducibles 2.000
     *          cuota = 累进(base) − 累进(mínimo personal 5.550) */
    tax: {
      type: 'marginal',
      gastosDeducibles: 2000,
      minimoPersonal: 5550,
      brackets: [
        [12450,    0.19],
        [20200,    0.24],
        [35200,    0.30],
        [60000,    0.37],
        [300000,   0.45],
        [Infinity, 0.47]
      ]
    },

    defaults: {
      grossAnnual: 40000, salaryMonths: 12, bonusCash: 0, allowanceMonthly: 0,
      stockAnnual: 0, stockType: 0, fundRate: 0, includeCompanyFund: false,
      afterTax: false, specialDeduct: 0, socialCap: 4909.5,
      workDaysPerWeek: 2, dailyHours: 8.5, commuteHours: 1, commuteComfort: 2,
      slackHours: 1.5, annualLeave: 23, leaveHard: 0, publicHolidays: 14, sickLeave: 0,
      city: 4, companyType: 1, age: 30, degree: 1, school: 1, bachelorSchool: 1, workYears: 2
    },

    weekDayOptions: [
      { v: 4,   label: '4 天（jornada de 4 días）' },
      { v: 4.5, label: '4.5 天' },
      { v: 5,   label: '5 天（jornada completa）' },
      { v: 5.5, label: '5.5 天' },
      { v: 6,   label: '6 天' }
    ],

    sources: [
      '基准时薪 €12：非马德里/巴塞的 junior 中位 ≈ €22,000 ÷ 年有效工作日 224 ÷ 日均有效工时 8.25',
      '薪资曲线：junior 20~30k（媒体 25k）· semi-senior 2~5 年 35~50k · senior 6~7 年以上常超 50k，大公司可到 70k+',
      '城市：Madrid 平均 48k、Cataluña 45k；马德里/巴塞相对全国有 18~22% 溢价',
      'Cárnica / consultora：马德里 senior 在中型咨询常只有 38~42k（无 bonus 无 equity）',
      'Seguridad Social 2025 劳工方 6.48%（4.70 常见风险 + 1.55 失业 + 0.10 培训 + 0.13 MEI），base máxima 4.909,50 €/月',
      'IRPF 2025 合并阶梯 19/24/30/37/45/47%；mínimo personal 5.550 €；gastos deducibles 2.000 €',
      '假期：法定最低 22 días laborables + 14 festivos'
    ]
  };

  var COUNTRIES = { CN: CN, ES: ES };
  var COUNTRY_LIST = [CN, ES];

  /* =========================================================================
   * 三、评级与雷达
   * ========================================================================= */
  var RATINGS = [
    { max: 0.45,     key: 'flee',   label: '建议立刻跑路', emoji: '🔥', color: '#e0413a',
      desc: '这份工作在系统性地消耗你。钱、时间、成长三样至少丢了两样，越待越难走。' },
    { max: 0.70,     key: 'bad',    label: '明显吃亏',     emoji: '😰', color: '#e8763a',
      desc: '你付出的和拿到的不成比例。除非有明确的短期目的（攒经验/等期权/过渡），否则该看机会了。' },
    { max: 0.95,     key: 'below',  label: '略低于市场',   emoji: '😐', color: '#d9a13a',
      desc: '不算惨，但你在用低于市场价的价格出售时间。调薪谈不下来的话，跳槽的收益会很直接。' },
    { max: 1.25,     key: 'median', label: '行业中位水平', emoji: '🙂', color: '#3a8fd9',
      desc: '典型的本地程序员处境：不算亏也不算赚。想往上走，先看诊断里最拖后腿的那一项。' },
    { max: 1.70,     key: 'good',   label: '相当不错',     emoji: '😀', color: '#2f9e6e',
      desc: '明显好于本地中位。多数人拿着比这差的条件在硬扛，你的位置值得守住。' },
    { max: 2.40,     key: 'great',  label: '神仙工作',     emoji: '🤩', color: '#7b4fd9',
      desc: '钱、时间、成长、环境至少有三项在水准之上。这种组合在市场上是稀缺品。' },
    { max: Infinity, key: 'dream',  label: '闷声发财',     emoji: '👑', color: '#c9971f',
      desc: '别在群里说。这个分数意味着你要么踩中了红利，要么谈判能力远超同侪。' }
  ];

  var RADAR_DIMS = [
    { key: 'pay',    label: '薪酬',   hint: '相对你的学历年限期望，钱给够了没有' },
    { key: 'time',   label: '时间',   hint: '每天被这份工作占掉多久' },
    { key: 'growth', label: '成长',   hint: '三年后你会更值钱还是更贬值' },
    { key: 'env',    label: '环境',   hint: '领导、同事、需求、设备' },
    { key: 'stable', label: '稳定',   hint: '公司类型、业务健康度、年龄风险' }
  ];

  /* =========================================================================
   * 四、表单构建（按国家）
   * ========================================================================= */
  function opts(list) {
    return list.map(function (o, i) { return { value: i, label: o.label, hint: o.hint }; });
  }
  function neutralIdx(options, fallback) {
    for (var i = 0; i < options.length; i++) if (options[i].v === 1.00) return i;
    return fallback === undefined ? 0 : fallback;
  }

  function buildSections(cc) {
    var C = COUNTRIES[cc] || CN;
    var d = C.defaults;
    var cur = C.cur;

    /* ---- 薪酬包 ---- */
    var payFields = [];
    if (C.salaryMode === 'monthly') {
      payFields.push(
        { key: 'monthlyBase', type: 'number', label: '月基本工资', unit: cur, def: d.monthlyBase, min: 0, step: 1000,
          hint: '税前 base，不含年终奖与股票' },
        { key: 'salaryMonths', type: 'number', label: '年薪月数', unit: '月', def: d.salaryMonths, min: 12, max: 30, step: 0.5,
          hint: '12 + 年终奖月数。13 薪填 13，年终 3 个月填 15' }
      );
    } else {
      payFields.push(
        { key: 'grossAnnual', type: 'number', label: '税前年薪（bruto anual）', unit: cur, def: d.grossAnnual, min: 0, step: 1000,
          hint: '不含 variable 与股票。12 或 14 pagas 不影响年总额，填全年总数即可' },
        { key: 'salaryMonths', type: 'number', label: 'Pagas（仅供参考）', unit: '期', def: d.salaryMonths, min: 12, max: 16, step: 1,
          hint: '只影响每月到手节奏，不影响年总额与得分' }
      );
    }
    payFields.push(
      { key: 'bonusCash', type: 'number',
        label: C.salaryMode === 'annual' ? '年度 variable / bonus' : '其他年度现金', unit: cur,
        def: d.bonusCash, min: 0, step: 1000,
        hint: C.salaryMode === 'annual' ? '按实际拿到的期望值填，不是「最高可达」' : '项目奖、签字费按年摊、专项激励等' },
      { key: 'allowanceMonthly', type: 'number',
        label: C.salaryMode === 'annual' ? '每月弹性薪酬 / 餐补' : '每月补贴合计', unit: cur,
        def: d.allowanceMonthly, min: 0, step: 50,
        hint: C.salaryMode === 'annual' ? 'Ticket restaurant、交通卡、guardería 等' : '餐补、交通补、通讯补、加班餐费' },
      { key: 'stockAnnual', type: 'number', label: '股票 / 期权年化面值', unit: cur, def: d.stockAnnual, min: 0, step: 1000,
        hint: '按 offer 上的授予总额 ÷ 归属年数' },
      { key: 'stockType', type: 'select', label: '股票兑现难度', def: d.stockType,
        options: opts(STOCK_DISCOUNT), hint: '面值 ≠ 到手，这一项决定折价多少' }
    );
    if (C.hasHousingFund) {
      payFields.push(
        { key: 'fundRate', type: 'select', label: '公积金比例', def: d.fundRate,
          options: [{ value: 0, label: '5%（最低档）' }, { value: 1, label: '7%' }, { value: 2, label: '8%' },
                    { value: 3, label: '10%' }, { value: 4, label: '12%（最高档）' }],
          hint: '公司和个人各按此比例缴纳' },
        { key: 'includeCompanyFund', type: 'toggle', label: '把公司缴的公积金计入总包', def: d.includeCompanyFund,
          hint: '这是实打实进你账户的钱，只是不能随便花' }
      );
    }
    payFields.push(
      { key: 'afterTax', type: 'toggle', label: '按税后口径计算', def: d.afterTax,
        hint: '扣除' + C.labels.taxName + '与个人社保。跨国比较务必打开 —— 西班牙税负明显重于中国，税前比较会高估西班牙 offer' },
      { key: 'specialDeduct', type: 'number',
        label: C.key === 'ES' ? '额外年度扣除（hijos / plan de pensiones…）' : '专项附加扣除',
        unit: C.key === 'ES' ? cur + '/年' : cur + '/月',
        def: d.specialDeduct, min: 0, step: C.key === 'ES' ? 500 : 500,
        hint: C.key === 'ES'
          ? '子女、赡养、养老金计划等额外减免的年度合计。留 0 即按单身无子女估算'
          : '房租房贷 / 子女教育 / 赡养老人 / 继续教育合计', dep: 'afterTax' },
      { key: 'socialCap', type: 'number', label: '社保缴费基数上限', unit: cur + '/月',
        def: d.socialCap, min: 1000, step: 100,
        hint: C.key === 'ES' ? '2025 年 base máxima = 4.909,50 €/月' : '各地不同，一线约 3.6 万', dep: 'afterTax' }
    );

    /* ---- 时间成本 ---- */
    var timeFields = [
      { key: 'workDaysPerWeek', type: 'select', label: '每周工作天数', def: d.workDaysPerWeek,
        options: C.weekDayOptions.map(function (o, i) { return { value: i, label: o.label }; }) },
      { key: 'dailyHours', type: 'number', label: '日均在司时长', unit: '小时', def: d.dailyHours, min: 1, max: 24, step: 0.5,
        hint: '下班时间 − 上班时间，含午休和吃饭，不含通勤' },
      { key: 'overtimeComp', type: 'select', label: '加班补偿', def: 0,
        options: opts(OVERTIME_COMP), hint: '只作用于超过 8 小时的部分' },
      { key: 'oncall', type: 'select', label: 'On-call / 值班强度（guardias）', def: 0,
        options: opts(ONCALL), hint: '程序员特有成本：不在公司也在工作' },
      { key: 'commuteHours', type: 'number', label: '每天通勤往返', unit: '小时', def: d.commuteHours, min: 0, max: 8, step: 0.25,
        hint: '家到公司 + 公司回家的总时长' },
      { key: 'commuteComfort', type: 'select', label: '通勤方式', def: d.commuteComfort,
        options: opts(COMMUTE_COMFORT), hint: '同样一小时，痛苦程度差很多' },
      { key: 'wfhDays', type: 'number', label: '每周远程办公天数', unit: '天', def: 0, min: 0, max: 7, step: 0.5,
        hint: '会按比例线性削减通勤成本' },
      { key: 'slackHours', type: 'number', label: '每天摸鱼 + 午休', unit: '小时', def: d.slackHours, min: 0, max: 12, step: 0.5,
        hint: '按 0.5 折回 —— 人还在公司，不能完全当没上班' },
      { key: 'annualLeave', type: 'number', label: '年假天数', unit: '天', def: d.annualLeave, min: 0, max: 60, step: 1,
        hint: C.labels.leaveHint },
      { key: 'leaveHard', type: 'select', label: '请假难度', def: d.leaveHard,
        options: opts(LEAVE_HARD), hint: '有年假 ≠ 能休年假' },
      { key: 'publicHolidays', type: 'number', label: '法定假日', unit: '天', def: d.publicHolidays, min: 0, max: 40, step: 1 },
      { key: 'sickLeave', type: 'number', label: '带薪病假', unit: '天', def: d.sickLeave, min: 0, max: 60, step: 1,
        hint: C.key === 'ES' ? '西班牙的 baja médica 不设上限，通常填 0' : '按 0.6 权重折算' }
    ];

    /* ---- 技术成长 ---- */
    var growthFields = GROWTH_DIMS.map(function (dim) {
      return { key: dim.key, type: 'select',
               label: dim.label + '（权重 ' + Math.round(dim.weight * 100) + '%）',
               def: neutralIdx(dim.options, 2), options: opts(dim.options), hint: dim.hint };
    });

    /* ---- 团队与环境 ---- */
    var envFields = ENV_DIMS.map(function (dim) {
      return { key: dim.key, type: 'select',
               label: dim.label + '（权重 ' + Math.round(dim.weight * 100) + '%）',
               def: neutralIdx(dim.options, 1), options: opts(dim.options), hint: dim.hint };
    }).concat([
      { key: 'city', type: 'select', label: '所在城市（按生活成本）', def: d.city,
        options: opts(C.cities), hint: '数值最低的城市最贵 —— 衡量的是钱经不经花，不是城市好不好' },
      { key: 'perks', type: 'checks', label: '福利加分（可多选）', def: [],
        options: PERKS.map(function (p) { return { value: p.key, label: p.label + '  +' + p.v.toFixed(3) }; }) }
    ]);

    /* ---- 公司与风险 ---- */
    var riskFields = [
      { key: 'companyType', type: 'select', label: '公司类型', def: d.companyType,
        options: C.companyTypes.map(function (c, i) { return { value: i, label: c.label, hint: c.hint }; }),
        hint: '同时决定「薪资涨幅期望」和「基础风险」两个系数' }
    ].concat(RISK_DIMS.map(function (dim) {
      return { key: dim.key, type: 'select', label: dim.label, def: 0, options: opts(dim.options) };
    })).concat([
      { key: 'contract', type: 'select', label: C.contractDim.label, def: 0, options: opts(C.contractDim.options) }
    ]);

    /* ---- 个人背景 ---- */
    var selfFields = [
      { key: 'age', type: 'number', label: '年龄', unit: '岁', def: d.age, min: 16, max: 70, step: 1,
        hint: C.key === 'ES'
          ? '西班牙没有中国式的 35 岁门槛，年龄惩罚只有中国的三分之一左右'
          : '用于计算年龄风险；体制内 / 国企 / 外企会大幅削减这一惩罚' },
      { key: 'degree', type: 'select', label: '最高学历', def: d.degree,
        options: C.degrees.map(function (x, i) { return { value: i, label: x.label }; }) },
      { key: 'school', type: 'select', label: '最高学历院校档次', def: d.school,
        options: C.schools.map(function (x, i) { return { value: i, label: x.label }; }) },
      { key: 'bachelorSchool', type: 'select', label: '本科院校档次', def: d.bachelorSchool,
        options: C.schools.map(function (x, i) { return { value: i, label: x.label }; }),
        hint: '仅硕士需要填 —— 硕士系数会追溯第一学历', dep: 'isMaster' },
      { key: 'workYears', type: 'select', label: '工作年限', def: d.workYears,
        options: C.years.map(function (y, i) { return { value: i, label: y.label }; }) }
    ];

    return [
      { id: 'comp',   title: C.labels.salaryTitle, icon: cur, desc: C.labels.salaryDesc, fields: payFields },
      { id: 'time',   title: '时间成本', icon: '⏱',
        desc: C.key === 'ES'
          ? '西班牙法定周工时 40 小时、年假最低 22 días laborables + 14 festivos。时间是唯一不可再生的成本。'
          : '2025 年中国程序员平均周工时 48.3 小时，仅 50% 能拿到加班补偿。时间是唯一不可再生的成本。',
        fields: timeFields },
      { id: 'growth', title: '技术成长', icon: '↗',
        desc: '一份工作除了给钱，还决定你三年后值多少钱。这一组按权重取加权平均，不连乘。', fields: growthFields },
      { id: 'env',    title: '团队与环境', icon: '◍',
        desc: 'Stack Overflow 2025 调查中，「自主权与信任」排在满意度因素第 1 位，高于薪酬。人的因素比想象中重要。',
        fields: envFields },
      { id: 'risk',   title: '公司与风险', icon: '⚠',
        desc: '风险是分母：同样的钱，越不稳定越不值。这一组用加法而不是乘法，避免叠加后分数塌缩。', fields: riskFields },
      { id: 'self',   title: '个人背景', icon: '◇',
        desc: '学历和年限都在分母。它们衡量的不是你有多强，而是市场本该给你多少 —— 期望越高，同样的薪水得分越低。',
        fields: selfFields }
    ];
  }

  root.PJC_DATA = {
    GLOBAL: GLOBAL,
    CONST: GLOBAL,                 // 兼容旧名
    STOCK_DISCOUNT: STOCK_DISCOUNT,
    OVERTIME_COMP: OVERTIME_COMP,
    COMMUTE_COMFORT: COMMUTE_COMFORT,
    ONCALL: ONCALL,
    LEAVE_HARD: LEAVE_HARD,
    GROWTH_DIMS: GROWTH_DIMS,
    ENV_DIMS: ENV_DIMS,
    PERKS: PERKS,
    RISK_DIMS: RISK_DIMS,
    COUNTRIES: COUNTRIES,
    COUNTRY_LIST: COUNTRY_LIST,
    RATINGS: RATINGS,
    RADAR_DIMS: RADAR_DIMS,
    buildSections: buildSections
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = root.PJC_DATA;

})(typeof window !== 'undefined' ? window : globalThis);
