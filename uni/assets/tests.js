/* ===========================================================================
 * tests.js —— 标定测试
 * ---------------------------------------------------------------------------
 * 两类检查：
 *   A. 不变量（invariants）—— 必须成立的数学性质，失败就是 bug
 *   B. 人群矩阵（personas）—— 真实场景的得分，靠人判断合不合常理
 *
 * Node:  node assets/tests.js
 * 浏览器: 打开 tests.html
 * =========================================================================== */

(function (root) {
  'use strict';

  var D = root.UNI_DATA  || (typeof require !== 'undefined' ? require('./data.js')  : null);
  var M = root.UNI_MODEL || (typeof require !== 'undefined' ? require('./model.js') : null);

  var f3 = function (n) { return (Math.round(n * 1000) / 1000).toFixed(3); };
  var f2 = function (n) { return (Math.round(n * 100) / 100).toFixed(2); };

  /* 用 key 而不是索引来构造 state —— 索引会随 data.js 调整而错位 */
  function idxOf(dims, key, label) {
    for (var i = 0; i < dims.length; i++) {
      if (dims[i].key !== key) continue;
      for (var j = 0; j < dims[i].options.length; j++) {
        if (dims[i].options[j].label.indexOf(label) === 0) return j;
      }
      throw new Error('找不到选项 ' + key + ' / ' + label);
    }
    throw new Error('找不到维度 ' + key);
  }
  function locIdx(key) {
    for (var i = 0; i < D.LOCATIONS.length; i++) if (D.LOCATIONS[i].key === key) return i;
    throw new Error('找不到地段 ' + key);
  }
  function regionIdx(cc, key) {
    var C = D.countryByKey(cc);
    for (var i = 0; i < C.regions.length; i++) if (C.regions[i].key === key) return i;
    throw new Error('找不到地区 ' + cc + '/' + key);
  }
  var ALLDIMS = [].concat(D.DORM_DIMS, D.CAMPUS_DIMS, D.PROSPECT_DIMS);
  function opt(key, labelPrefix) { return idxOf(ALLDIMS, key, labelPrefix); }

  /* 换地区时，花销也要重置到该地区的基准 —— 否则「全中位」只在默认地区成立，
   * 其余地区会带着上一个地区的钱数，成本比莫名其妙地偏掉。 */
  function build(cc, regionKey, patch) {
    var s = M.defaultState(cc);
    var i = regionIdx(cc, regionKey);
    var rg = D.countryByKey(cc).regions[i];
    s.region = i;
    s.monthlyLiving = rg.baseLiving;
    s.monthlyHousing = rg.baseHousing;
    s.monthlyOther = 0;
    return M.assign(s, patch || {});
  }

  /* =======================================================================
   * A. 不变量
   * ===================================================================== */
  var invariants = [];
  function inv(name, fn) { invariants.push({ name: name, fn: fn }); }

  inv('全中位 = 1.000', function () {
    var r = M.compute(M.defaultState('CN'));
    return Math.abs(r.score - 1) < 1e-9 ? null : '得到 ' + f3(r.score);
  });

  inv('每个国家的默认状态都是有限数且为正', function () {
    var bad = [];
    D.COUNTRIES.forEach(function (C) {
      C.regions.forEach(function (rg, i) {
        var s = M.defaultState(C.key); s.region = i;
        var r = M.compute(s);
        if (!isFinite(r.score) || r.score <= 0) bad.push(C.key + '/' + rg.key + '=' + r.score);
      });
    });
    return bad.length ? bad.join(', ') : null;
  });

  inv('花销单调：多花钱得分必降', function () {
    var lo = M.compute(build('CN', 'tier2', { monthlyLiving: 1000 })).score;
    var hi = M.compute(build('CN', 'tier2', { monthlyLiving: 4000 })).score;
    return hi < lo ? null : '1000→' + f3(lo) + ' 4000→' + f3(hi);
  });

  inv('住宿单调：每个维度调好都不该降分', function () {
    var bad = [];
    D.DORM_DIMS.concat(D.CAMPUS_DIMS).forEach(function (d) {
      var best = 0, bv = -Infinity;
      d.options.forEach(function (o, i) { if (o.v > bv) { bv = o.v; best = i; } });
      var worst = 0, wv = Infinity;
      d.options.forEach(function (o, i) { if (o.v < wv) { wv = o.v; worst = i; } });
      var p1 = {}, p2 = {}; p1[d.key] = worst; p2[d.key] = best;
      var a = M.compute(build('CN', 'tier2', p1)).score;
      var b = M.compute(build('CN', 'tier2', p2)).score;
      if (!(b > a)) bad.push(d.key + ' 差=' + f3(a) + ' 好=' + f3(b));
    });
    return bad.length ? bad.join('; ') : null;
  });

  inv('地段单调：市中心 > 远郊（同等花销同等条件）', function () {
    var core = M.compute(build('CN', 'tier2', { location: locIdx('core') })).score;
    var rem  = M.compute(build('CN', 'tier2', { location: locIdx('remote') })).score;
    // 注意：远郊的基准花销更低，所以同样花销下远郊的成本比更高（更亏）
    return core > rem ? null : '市中心=' + f3(core) + ' 远郊=' + f3(rem);
  });

  inv('实习时间惩罚：同薪同岗，通勤越长得分越低', function () {
    var p = { internQuality: 3, internDaysPerWeek: 5, internHours: 8, internDailyPay: 200, convertProb: 3 };
    var a = M.compute(build('CN', 'tier2', M.assign(p, { internCommute: 0.5 }))).score;
    var b = M.compute(build('CN', 'tier2', M.assign(p, { internCommute: 4 }))).score;
    return a > b ? null : '通勤0.5h=' + f3(a) + ' 4h=' + f3(b);
  });

  inv('实习不该产生负的净支出', function () {
    var r = M.compute(build('CN', 'tier2',
      { internQuality: 4, internDaysPerWeek: 6, internHours: 8, internDailyPay: 800, convertProb: 5 }));
    return r.netMonthly >= 0 ? null : '净支出 ' + r.netMonthly;
  });

  inv('转正概率单调：概率越高得分越高', function () {
    var base = { internQuality: 3, internDaysPerWeek: 3, internHours: 8, internCommute: 1, internDailyPay: 200 };
    var prev = -Infinity, bad = null;
    for (var i = 0; i < D.CONVERT_PROB.length; i++) {
      var sc = M.compute(build('CN', 'tier2', M.assign(base, { convertProb: i }))).score;
      if (sc < prev - 1e-9) bad = '在第 ' + i + ' 档下降';
      prev = sc;
    }
    return bad;
  });

  inv('PPP 中性：同国同条件下，得分与币种无关（分子分母同币种约掉）', function () {
    // 同一处境用「当地基准」表述，各国得分应当一致
    var bad = [];
    D.COUNTRIES.forEach(function (C) {
      var rg = C.regions[0];
      var s = M.defaultState(C.key);
      s.region = 0;
      s.monthlyLiving = rg.baseLiving;
      s.monthlyHousing = rg.baseHousing;
      s.monthlyOther = 0;
      var sc = M.compute(s).score;
      if (Math.abs(sc - 1) > 1e-9) bad.push(C.key + '=' + f3(sc));
    });
    return bad.length ? bad.join(', ') : null;
  });

  inv('雷达五维都落在 0~100', function () {
    var bad = [];
    [build('CN', 'tier1', { monthlyLiving: 100 }),
     build('CN', 'county', { monthlyLiving: 99999 }),
     build('CN', 'tier2', {}),
     build('US', 'usmajor', { internQuality: 4, internDaysPerWeek: 6, internHours: 12, internDailyPay: 400 })
    ].forEach(function (s, i) {
      var rd = M.compute(s).radar;
      Object.keys(rd).forEach(function (k) {
        if (!(rd[k] >= 0 && rd[k] <= 100)) bad.push('#' + i + ' ' + k + '=' + rd[k]);
      });
    });
    return bad.length ? bad.join(', ') : null;
  });

  inv('零花钱填 0 不产生 NaN / Infinity', function () {
    var r = M.compute(build('CN', 'tier2', { monthlyLiving: 0, monthlyHousing: 0, monthlyOther: 0 }));
    return isFinite(r.score) ? null : '得分 ' + r.score;
  });

  inv('诊断只给出真正能改的项', function () {
    var rows = M.diagnose(build('CN', 'tier2', {}));
    var bad = rows.filter(function (r) {
      return r.key === 'major' || r.key === 'schoolTier' || r.key === 'localJobs';
    });
    return bad.length ? '出现了不可改项: ' + bad.map(function (b) { return b.key; }).join(',') : null;
  });

  /* =======================================================================
   * B. 人群矩阵
   * ===================================================================== */
  var PERSONAS = [
    /* ---------- 国内 · 一线 ---------- */
    { g: '国内一线', name: '北京985计算机·市区·四人间·家里给3000·大厂实习3天',
      s: build('CN', 'tier1', {
        monthlyLiving: 2600, monthlyHousing: 200, monthlyOther: 200,
        location: locIdx('urban'), commute: 0,
        roomSize: opt('roomSize', '四人间'), bathroom: opt('bathroom', '楼层公共卫生间'),
        shower: opt('shower', '楼层公共浴室'), power: opt('power', '深夜断电'),
        roommate: opt('roommate', '基本和谐'), climate: opt('climate', '有独立空调'),
        major: opt('major', '计算机'), schoolTier: opt('schoolTier', '985'),
        localJobs: opt('localJobs', '本地就是'), internship: opt('internship', '大二起'),
        internQuality: 4, internDaysPerWeek: 3, internHours: 9, internCommute: 1.5,
        internDailyPay: 250, convertProb: 3 }) },

    { g: '国内一线', name: '上海211金融·市中心·双人间·家里给6000·无实习',
      s: build('CN', 'tier1c', {
        monthlyLiving: 4500, monthlyHousing: 1500, monthlyOther: 300,
        location: locIdx('core'), commute: 0,
        roomSize: opt('roomSize', '双人间'), bathroom: opt('bathroom', '房间内独立'),
        shower: opt('shower', '房间内独立'), power: opt('power', '24 小时'),
        roommate: opt('roommate', '基本和谐'), housing: opt('housing', '校外合租'),
        major: opt('major', '金融'), schoolTier: opt('schoolTier', '211'),
        localJobs: opt('localJobs', '本地就是') }) },

    { g: '国内一线', name: '深圳普通一本·远郊校区·六人间·家里给1500·无实习',
      s: build('CN', 'tier1', {
        monthlyLiving: 1400, monthlyHousing: 100,
        location: locIdx('remote'), commute: 3,
        roomSize: opt('roomSize', '六人间'), shower: opt('shower', '公共澡堂，定时'),
        power: opt('power', '断电且断网'), curfew: opt('curfew', '22:30'),
        roommate: opt('roommate', '各过各的'), climate: opt('climate', '只有风扇'),
        major: opt('major', '经管'), schoolTier: opt('schoolTier', '普通一本'),
        localJobs: opt('localJobs', '基本要去') }) },

    /* ---------- 国内 · 新一线 / 二线 ---------- */
    { g: '国内新一线', name: '武汉985电子·大学城·四人间·2000·一般实习2天',
      s: build('CN', 'newt1', {
        monthlyLiving: 1800, monthlyHousing: 180, location: locIdx('unitown'), commute: 1,
        roomSize: opt('roomSize', '四人间'), major: opt('major', '电子信息'),
        schoolTier: opt('schoolTier', '985'), localJobs: opt('localJobs', '本地有成规模'),
        internQuality: 2, internDaysPerWeek: 2, internHours: 8, internCommute: 1.5,
        internDailyPay: 120, convertProb: 2 }) },

    { g: '国内二线', name: '南昌二本农学·城郊·六人间·1200·无实习（典型天坑）',
      s: build('CN', 'tier3', {
        monthlyLiving: 1100, monthlyHousing: 120, location: locIdx('fringe'), commute: 3,
        roomSize: opt('roomSize', '六人间'), shower: opt('shower', '楼层公共浴室'),
        power: opt('power', '断电且断网'), curfew: opt('curfew', '22:30'),
        strictness: opt('strictness', '强制早操'), roommate: opt('roommate', '各过各的'),
        major: opt('major', '农林'), schoolTier: opt('schoolTier', '普通二本'),
        localJobs: opt('localJobs', '本地几乎没有'), internship: opt('internship', '学校不支持') }) },

    /* ---------- 国内 · 三线 / 县城 ---------- */
    { g: '国内三线', name: '县城专科·八人间·800·无实习（最低配）',
      s: build('CN', 'county', {
        monthlyLiving: 750, monthlyHousing: 80, location: locIdx('unitown'), commute: 3,
        roomSize: opt('roomSize', '八人及以上'), bathroom: opt('bathroom', '需要出楼'),
        shower: opt('shower', '需走出宿舍楼'), power: opt('power', '断得早'),
        curfew: opt('curfew', '22:00 前'), strictness: opt('strictness', '早操 + 晚自习'),
        roommate: opt('roommate', '有明显摩擦'), climate: opt('climate', '什么都没有'),
        canteen: opt('canteen', '选择极少'), facility: opt('facility', '老旧'),
        major: opt('major', '旅游'), schoolTier: opt('schoolTier', '专科'),
        localJobs: opt('localJobs', '本地几乎没有') }) },

    { g: '国内三线', name: '三线师范·城郊·四人间·1300·家教兼职（低成本还行）',
      s: build('CN', 'tier3', {
        monthlyLiving: 1150, monthlyHousing: 120, location: locIdx('fringe'), commute: 1,
        roomSize: opt('roomSize', '四人间'), roommate: opt('roommate', '作息一致'),
        major: opt('major', '师范'), schoolTier: opt('schoolTier', '普通一本'),
        localJobs: opt('localJobs', '本地有一些'),
        internQuality: 1, internRelevance: 1, internTerm: 0,
        internDaysPerWeek: 2, internHours: 3, internCommute: 0.5,
        internDailyPay: 150, convertProb: 0 }) },

    /* ---------- 极端花销 ---------- */
    { g: '极端', name: '二线·条件普通但每月花8000（富裕但不划算）',
      s: build('CN', 'tier2', { monthlyLiving: 7000, monthlyHousing: 1000 }) },
    { g: '极端', name: '二线·条件普通但每月只花700（极省）',
      s: build('CN', 'tier2', { monthlyLiving: 600, monthlyHousing: 100 }) },
    { g: '极端', name: '全项最优·无实习',
      s: build('CN', 'tier2', (function () {
        var p = { location: locIdx('core'), commute: 0, monthlyLiving: 1400, monthlyHousing: 200 };
        ALLDIMS.forEach(function (d) {
          var b = 0, bv = -Infinity;
          d.options.forEach(function (o, i) { if (o.v > bv) { bv = o.v; b = i; } });
          p[d.key] = b;
        });
        return p;
      })()) },
    { g: '极端', name: '全项最差·无实习',
      s: build('CN', 'tier2', (function () {
        var p = { location: locIdx('remote'), commute: 4, monthlyLiving: 2200, monthlyHousing: 300 };
        ALLDIMS.forEach(function (d) {
          var w = 0, wv = Infinity;
          d.options.forEach(function (o, i) { if (o.v < wv) { wv = o.v; w = i; } });
          p[d.key] = w;
        });
        return p;
      })()) },

    /* ---------- 境外 ---------- */
    { g: '境外欧洲', name: '巴塞罗那授课硕士·市区合租·€1000/月·无实习',
      s: build('ES', 'bcnmad', {
        stage: 1, monthlyLiving: 450, monthlyHousing: 550, location: locIdx('urban'), commute: 0,
        housing: opt('housing', '校外合租'), roomSize: opt('roomSize', '单人间'),
        bathroom: opt('bathroom', '宿舍单元内共用'), shower: opt('shower', '宿舍单元内共用'),
        power: opt('power', '24 小时'), curfew: opt('curfew', '无门禁'),
        strictness: opt('strictness', '完全自主'), roommate: opt('roommate', '各过各的'),
        major: opt('major', '计算机'), schoolTier: opt('schoolTier', '双一流学科'),
        localJobs: opt('localJobs', '本地有成规模') }) },

    { g: '境外欧洲', name: '西班牙小城本科·大学城·€600/月·超省',
      s: build('ES', 'essmall', {
        monthlyLiving: 300, monthlyHousing: 300, location: locIdx('unitown'), commute: 1,
        housing: opt('housing', '校外合租'), roomSize: opt('roomSize', '单人间'),
        power: opt('power', '24 小时'), curfew: opt('curfew', '无门禁'),
        strictness: opt('strictness', '完全自主'),
        major: opt('major', '外语'), schoolTier: opt('schoolTier', '普通一本'),
        localJobs: opt('localJobs', '基本要去') }) },

    { g: '境外欧洲', name: '伦敦授课硕士·市中心·£1600/月·无实习（最贵场景）',
      s: build('UK', 'london', {
        stage: 1, monthlyLiving: 600, monthlyHousing: 1000, location: locIdx('core'), commute: 0,
        housing: opt('housing', '校外合租'), roomSize: opt('roomSize', '单人间'),
        bathroom: opt('bathroom', '宿舍单元内共用'), power: opt('power', '24 小时'),
        curfew: opt('curfew', '无门禁'), strictness: opt('strictness', '完全自主'),
        major: opt('major', '金融'), schoolTier: opt('schoolTier', '985 / QS 51-100'),
        localJobs: opt('localJobs', '本地就是') }) },

    { g: '境外北美', name: '美国大学城本科·$1400/月·校内宿舍双人间',
      s: build('US', 'ustown', {
        monthlyLiving: 620, monthlyHousing: 780, location: locIdx('unitown'), commute: 2,
        roomSize: opt('roomSize', '双人间'), power: opt('power', '24 小时'),
        curfew: opt('curfew', '无门禁'), strictness: opt('strictness', '完全自主'),
        canteen: opt('canteen', '正常水平'),
        major: opt('major', '计算机'), schoolTier: opt('schoolTier', '双一流学科'),
        localJobs: opt('localJobs', '基本要去') }) },

    { g: '境外亚洲', name: '东京授课硕士·地方城市·¥JP95000/月·便利店兼职',
      s: build('JP', 'jpsmall', {
        stage: 1, monthlyLiving: 55000, monthlyHousing: 40000,
        location: locIdx('urban'), commute: 0,
        housing: opt('housing', '校外整租'), roomSize: opt('roomSize', '单人间'),
        bathroom: opt('bathroom', '房间内独立'), shower: opt('shower', '房间内独立'),
        power: opt('power', '24 小时'), curfew: opt('curfew', '无门禁'),
        strictness: opt('strictness', '完全自主'),
        major: opt('major', '电子信息'), schoolTier: opt('schoolTier', '双一流学科'),
        localJobs: opt('localJobs', '本地有一些'),
        internQuality: 1, internRelevance: 0, internTerm: 0,
        internDaysPerWeek: 3, internHours: 5, internCommute: 0.5,
        internDailyPay: 6000, convertProb: 0 }) },

    { g: '境外亚洲', name: '新加坡QS前50计算机·S$1900/月·大厂实习',
      s: build('SG', 'sg', {
        monthlyLiving: 750, monthlyHousing: 1150, location: locIdx('urban'), commute: 0,
        roomSize: opt('roomSize', '双人间'), power: opt('power', '24 小时'),
        curfew: opt('curfew', '无门禁'), strictness: opt('strictness', '完全自主'),
        major: opt('major', '计算机'), schoolTier: opt('schoolTier', 'C9'),
        localJobs: opt('localJobs', '本地就是'), internship: opt('internship', '大二起'),
        internQuality: 4, internDaysPerWeek: 5, internHours: 8, internCommute: 1,
        internDailyPay: 90, convertProb: 4 }) },

    { g: '境外亚洲', name: '香港授课硕士·九龙市区·HK$11000/月（贵且住得挤）',
      s: build('HK', 'hkurban', {
        stage: 1, monthlyLiving: 4500, monthlyHousing: 6500,
        location: locIdx('core'), commute: 0,
        housing: opt('housing', '校外合租'), roomSize: opt('roomSize', '双人间'),
        power: opt('power', '24 小时'), curfew: opt('curfew', '无门禁'),
        strictness: opt('strictness', '完全自主'),
        major: opt('major', '金融'), schoolTier: opt('schoolTier', '211'),
        localJobs: opt('localJobs', '本地就是') }) },

    { g: '境外亚洲', name: '泰国曼谷本科·฿20000/月·物价低',
      s: build('TH', 'thbkk', {
        monthlyLiving: 12000, monthlyHousing: 8000, location: locIdx('urban'), commute: 1,
        roomSize: opt('roomSize', '单人间'), bathroom: opt('bathroom', '房间内独立'),
        power: opt('power', '24 小时'), curfew: opt('curfew', '无门禁'),
        major: opt('major', '经管'), schoolTier: opt('schoolTier', '普通一本'),
        localJobs: opt('localJobs', '本地有一些') }) },

    /* ---------- 实习分支 ---------- */
    { g: '实习分支', name: '二线·无实习（对照组）', s: build('CN', 'tier2', {}) },
    { g: '实习分支', name: '二线·轻度实习2天150/天',
      s: build('CN', 'tier2', { internQuality: 2, internDaysPerWeek: 2, internHours: 8,
                                internCommute: 1, internDailyPay: 150, convertProb: 2 }) },
    { g: '实习分支', name: '二线·全职实习5天250/天·通勤2h·转正65%',
      s: build('CN', 'tier2', { internQuality: 3, internDaysPerWeek: 5, internHours: 9,
                                internCommute: 2, internDailyPay: 250, convertProb: 4 }) },
    { g: '实习分支', name: '二线·压榨实习6天300/天·12h+3h通勤·无转正',
      s: build('CN', 'tier2', { internQuality: 3, internDaysPerWeek: 6, internHours: 12,
                                internCommute: 3, internDailyPay: 300, convertProb: 1 }) },
    { g: '实习分支', name: '二线·已拿转正offer·3天大厂',
      s: build('CN', 'tier2', { internQuality: 4, internDaysPerWeek: 3, internHours: 8,
                                internCommute: 1, internDailyPay: 250, convertProb: 5 }) }
  ];

  /* =======================================================================
   * 运行
   * ===================================================================== */
  function run() {
    var invRows = invariants.map(function (t) {
      var msg = null, err = null;
      try { msg = t.fn(); } catch (e) { err = e.message; }
      return { name: t.name, pass: !msg && !err, detail: err || msg };
    });

    var perRows = PERSONAS.map(function (p) {
      var r = M.compute(p.s);
      return {
        group: p.g, name: p.name, score: r.score, rating: r.rating.label, emoji: r.rating.emoji,
        cur: r.cur,
        gross: r.grossMonthly, net: r.netMonthly, baseline: r.baselineMonthly,
        pppNet: r.pppNetMonthly, pppGross: r.pppGrossMonthly,
        costRatio: r.costRatio, floored: r.costFloored,
        dorm: r.dormFactor, loc: r.locationFactor, campus: r.campusFactor, prospect: r.prospectFactor,
        internIncome: r.intern.monthlyIncome, weeklyHours: r.intern.weeklyHours,
        timePenalty: r.intern.timePenalty,
        radar: r.radar,
        top: M.diagnose(p.s, 2).map(function (d) { return d.label + '(' + M.fmt.pct(d.pct) + ')'; }).join(' / ')
      };
    });

    return { invariants: invRows, personas: perRows };
  }

  var API = { run: run, PERSONAS: PERSONAS, invariants: invariants, f3: f3, f2: f2 };
  root.UNI_TESTS = API;
  if (typeof module !== 'undefined' && module.exports) module.exports = API;

  /* Node 直跑 */
  if (typeof require !== 'undefined' && typeof module !== 'undefined' && require.main === module) {
    var out = run();
    var fail = 0;
    console.log('\n══════ A. 不变量 ══════');
    out.invariants.forEach(function (t) {
      if (!t.pass) fail++;
      console.log((t.pass ? '  ✓ ' : '  ✗ ') + t.name + (t.detail ? '  → ' + t.detail : ''));
    });

    console.log('\n══════ B. 人群矩阵 ══════');
    var g = null;
    out.personas.forEach(function (p) {
      if (p.group !== g) { g = p.group; console.log('\n── ' + g + ' ──'); }
      console.log('  ' + f3(p.score).padStart(6) + ' ' + p.emoji + ' ' + p.rating.padEnd(7) + ' │ ' + p.name);
      console.log('        住' + f2(p.dorm) + ' 段' + f2(p.loc) + ' 校' + f2(p.campus) + ' 景' + f2(p.prospect) +
                  ' 成本比' + f2(p.costRatio) + (p.floored ? '(触底)' : '') +
                  ' │ 月支出 ' + p.cur + Math.round(p.gross) +
                  ' 净 ' + p.cur + Math.round(p.net) +
                  ' 基准 ' + p.cur + Math.round(p.baseline) +
                  ' │ 折人民币购买力 ¥' + Math.round(p.pppNet));
      if (p.internIncome > 0)
        console.log('        实习 周' + p.weeklyHours + 'h 月入' + p.cur + Math.round(p.internIncome) +
                    ' 时间×' + f2(p.timePenalty));
    });

    console.log('\n══════ 汇总 ══════');
    var scores = out.personas.map(function (p) { return p.score; });
    console.log('  不变量 ' + (out.invariants.length - fail) + '/' + out.invariants.length + ' 通过');
    console.log('  得分范围 ' + f3(Math.min.apply(null, scores)) + ' ~ ' + f3(Math.max.apply(null, scores)));
    process.exit(fail ? 1 : 0);
  }

})(typeof window !== 'undefined' ? window : globalThis);
