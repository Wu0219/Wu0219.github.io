/* ===========================================================================
 * model.js —— 计算引擎（纯函数，无 DOM 依赖，可在 Node 里直接跑）
 * ---------------------------------------------------------------------------
 *                        住宿系数 × 地段系数 × 校园系数 × 前景系数
 *   大学生活性价比  =  ──────────────────────────────────────────────
 *                                    净成本比
 *
 *   净成本比 = （生活费 + 住宿 + 其他 − 实习收入）÷ 当地同类学生基准花销
 *
 *   得分 1.00 = 在**当地**花这个钱，过上了这个价位应有的大学生活
 *
 *   为什么分母用「当地基准」而不是统一的人民币：
 *     性价比是相对概念。在巴塞罗那花 €800 和在南昌花 ¥1300，绝对值差 4 倍，
 *     但相对当地水平可能一样。分子分母同币种，汇率与 PPP 自然约掉。
 *     PPP 只用于把结果**显示**成人民币购买力，以及跨校横向比较。
 *
 *   适用：本科生 / 授课型硕士。博士与研究型硕士不适用。
 * =========================================================================== */

(function (root) {
  'use strict';

  var D = root.UNI_DATA || (typeof require !== 'undefined' ? require('./data.js') : null);
  var G = D.GLOBAL;

  /* ---------------- 小工具 ---------------- */
  function clamp(x, lo, hi) { return Math.min(Math.max(x, lo), hi); }
  function num(x, d) { var v = parseFloat(x); return isFinite(v) ? v : (d || 0); }
  function assign(a, b) { var o = {}, k; for (k in a) o[k] = a[k]; for (k in b) o[k] = b[k]; return o; }
  function pick(arr, i) { return arr[clamp(i | 0, 0, arr.length - 1)]; }

  /* ---------------- 默认状态 ---------------- */
  function defaultState(cc) {
    cc = cc || 'CN';
    var s = { country: cc };
    D.buildSections(cc).forEach(function (sec) {
      sec.fields.forEach(function (f) { s[f.key] = f.def; });
    });
    return s;
  }

  /* 换国家：钱和地区必须重置（币种变了），主观评价沿用 */
  var COUNTRY_SPECIFIC = ['region', 'monthlyLiving', 'monthlyHousing', 'monthlyOther', 'internDailyPay'];
  function switchCountry(state, cc) {
    var fresh = defaultState(cc);
    var out = assign(fresh, {});
    Object.keys(state).forEach(function (k) {
      if (k === 'country') return;
      if (COUNTRY_SPECIFIC.indexOf(k) >= 0) return;
      if (fresh.hasOwnProperty(k)) out[k] = state[k];
    });
    out.country = cc;
    return out;
  }

  /* ---------------- 加权平均（中位 = 1.00） ---------------- */
  function weightedFactor(dims, s) {
    var tw = 0, acc = 0;
    dims.forEach(function (d) {
      var o = pick(d.options, num(s[d.key], 0));
      acc += d.weight * o.v; tw += d.weight;
    });
    return tw ? acc / tw : 1;
  }
  function weightedRange(dims) {
    var tw = 0, hi = 0, lo = 0;
    dims.forEach(function (d) {
      var vs = d.options.map(function (o) { return o.v; });
      hi += d.weight * Math.max.apply(null, vs);
      lo += d.weight * Math.min.apply(null, vs);
      tw += d.weight;
    });
    return { min: lo / tw, max: hi / tw };
  }
  var DORM_RANGE     = weightedRange(D.DORM_DIMS);
  var CAMPUS_RANGE   = weightedRange(D.CAMPUS_DIMS);
  var PROSPECT_RANGE = weightedRange(D.PROSPECT_DIMS);

  /* ---------------- 实习 ---------------- */
  function internOf(s, C) {
    var q      = pick(D.INTERN_QUALITY,   num(s.internQuality, 0));
    var conv   = pick(D.CONVERT_PROB,     num(s.convertProb, 0));
    var rel    = pick(D.INTERN_RELEVANCE, num(s.internRelevance, 2));
    var term   = pick(D.INTERN_TERM,      num(s.internTerm, 0));
    var days   = clamp(num(s.internDaysPerWeek, 0), 0, 7);
    var hours  = clamp(num(s.internHours, 0), 0, 16);
    var commute= clamp(num(s.internCommute, 0), 0, 8);
    var pay    = Math.max(num(s.internDailyPay, 0), 0);

    var hasIntern = days > 0 && (num(s.internQuality, 0) > 0 || pay > 0);
    var weeklyHours = (hours + commute) * days;
    var grossIncome = hasIntern ? pay * days * D.WEEKS_PER_MONTH : 0;
    // 非对口零工只解决现金流、不积累职业资本，抵扣权重打折
    var creditedIncome = grossIncome * rel.v;

    /* 分段时间惩罚：把「正常全职」和「过劳」分开。
     * 单段线性会把 50h（全职+通勤）和 90h（压榨）压进同一个地板，
     * 于是选全职必然比兼职低分 —— 那是 bug 不是设计。 */
    var T = D.INTERN_TIME, pen = 0;
    if (weeklyHours > T.FREE) {
      if (weeklyHours <= T.NORMAL) {
        pen = (weeklyHours - T.FREE) / (T.NORMAL - T.FREE) * T.NORMAL_PENALTY;
      } else {
        pen = T.NORMAL_PENALTY +
              clamp((weeklyHours - T.NORMAL) / (T.SEVERE - T.NORMAL), 0, 1) *
              (T.MAX_PENALTY - T.NORMAL_PENALTY);
      }
    }
    pen *= term.v;                       // 假期实习几乎不占用课程时间
    var timePenalty = 1 - pen;

    // 含金量与转正概率相关，转正的边际贡献随含金量升高而衰减，并整体封顶
    var convertBonus = D.convertBonusOf(q.v, conv.v);
    var prospectBoost = hasIntern
      ? Math.min(q.v * convertBonus, D.PROSPECT_BOOST_CAP) : 1;

    return {
      has: hasIntern, quality: q, convert: conv, relevance: rel, term: term,
      daysPerWeek: days, hoursPerDay: hours, commutePerDay: commute,
      weeklyHours: weeklyHours, dailyPay: pay,
      grossIncome: grossIncome, monthlyIncome: grossIncome,
      creditedIncome: creditedIncome,
      hourlyPay: hours > 0 ? pay / hours : 0,
      hourlyPayWithCommute: (hours + commute) > 0 ? pay / (hours + commute) : 0,
      timePenalty: timePenalty, convertBonus: convertBonus, prospectBoost: prospectBoost
    };
  }

  /* =========================================================================
   * 主计算
   * ========================================================================= */
  function compute(s) {
    var C = D.countryByKey(s.country);
    var region = pick(C.regions, num(s.region, 0));
    var loc    = pick(D.LOCATIONS, num(s.location, 2));
    var comm   = pick(D.COMMUTE,   num(s.commute, 1));

    /* 1. 成本
     * 基准要和用户的居住形式对齐：住宿舍就比宿舍价，租房就比当地租房行情。
     * 用单一宿舍基准去衡量租房的学生，会让他永远显得「花超了」。
     * 地段溢价只作用于住宿，生活费只做很小幅度浮动 —— 住市中心还是郊区，
     * 地铁月票和超市价格并不会差 40%。 */
    var hb = D.HOUSING_BASE[clamp(num(s.housing, 3) | 0, 0, D.HOUSING_BASE.length - 1)];
    var baseHouseRaw = (hb.base === 'rent' ? (region.baseRent || region.baseHousing)
                                           : region.baseHousing) * hb.k;
    var baselineMonthly = region.baseLiving * loc.livingIdx + baseHouseRaw * loc.houseIdx;

    var grossMonthly = num(s.monthlyLiving) + num(s.monthlyHousing) + num(s.monthlyOther);
    var intern = internOf(s, C);

    /* 绝对地板：食物、通讯、日用这些刚性开支不可能被实习收入抵扣掉。
     * 用它做除零保护，比比例地板更有物理意义，饱和点也推得更远。 */
    var rigidFloor = baselineMonthly * G.RIGID_RATIO;
    var offset = Math.min(intern.creditedIncome, Math.max(grossMonthly - rigidFloor, 0));
    var netMonthly = grossMonthly - offset;
    var surplus = Math.max(intern.creditedIncome - offset, 0);

    var costRatioRaw = baselineMonthly > 0 ? netMonthly / baselineMonthly : 1;
    // 两端同时做幂压缩，避免「省钱奖励封顶、超支惩罚敞口」的不对称
    var costRatio = Math.pow(Math.min(costRatioRaw, G.COST_MAX), G.COST_EXP);
    var costFloored = offset > 0 && netMonthly <= rigidFloor + 1e-9;

    // 多赚的钱走对数加成：跨两三个数量级仍有分辨率，又不会发散
    var surplusBonus = 1 + Math.min(
      G.SURPLUS_MAX_BONUS,
      G.SURPLUS_COEF * Math.log(1 + surplus / Math.max(baselineMonthly, 1))
    );

    /* 2. 四个系数 */
    var dormFactor     = weightedFactor(D.DORM_DIMS, s);
    var campusRaw      = weightedFactor(D.CAMPUS_DIMS, s);
    var campusFactor   = campusRaw * intern.timePenalty;
    var locationFactor = loc.v * comm.v;
    var prospectRaw    = weightedFactor(D.PROSPECT_DIMS, s);
    var prospectFactor = prospectRaw * intern.prospectBoost;

    /* 3. 三个分数
     *   客观性价比 —— 钱换来了什么，可被外人核对
     *   主观体验   —— 你自己过得怎么样，别人无从置喙
     *   总分       —— 两者加权几何平均，客观占 6 成
     *
     * 分开算是因为这两件事真的可以背道而驰：条件很差但过得开心的人，
     * 和名校里长期抑郁的人，揉进一个数字两边都会失真。
     *
     * 四个系数直接连乘会四重复合（各偏 15% → 结果偏 75%），实测跨度 26 倍、
     * 中位数被推到 1.4。统一加压缩指数，排序不变但跨度收回可解释范围。 */
    var qualityRaw = dormFactor * locationFactor * campusFactor * prospectFactor;
    var quality = Math.pow(qualityRaw, G.QUALITY_EXP);
    var numerator = quality * surplusBonus;
    var objective = costRatio > 0 ? numerator / costRatio : 0;

    var subjRaw = weightedFactor(D.SUBJECTIVE_DIMS, s);
    var subjective = Math.pow(subjRaw, D.SUBJ_EXP);

    var score = objective * Math.pow(Math.max(subjective, 1e-6), D.SUBJ_WEIGHT);

    /* 4. PPP 折算（仅用于显示与跨国比较，不影响得分） */
    var pppMul = G.PPP_ANCHOR / C.ppp;
    var M = G.MONTHS_PER_YEAR;

    return {
      country: C, cur: C.cur, region: region, loc: loc, comm: comm,
      score: score, rating: ratingOf(score),
      objective: objective, objRating: ratingOf(objective),
      subjective: subjective, subjRaw: subjRaw, subjRating: ratingOf(subjective),

      grossMonthly: grossMonthly, netMonthly: netMonthly,
      internOffset: offset, surplus: surplus, surplusBonus: surplusBonus,
      rigidFloor: rigidFloor,
      selfSufficient: costFloored,
      qualityRaw: qualityRaw, quality: quality,
      baselineHousing: baseHouseRaw, baselineMonthly: baselineMonthly,
      costRatio: costRatio, costRatioRaw: costRatioRaw, costFloored: costFloored,
      annualGross: grossMonthly * M, annualNet: netMonthly * M,

      intern: intern,
      internCoverage: grossMonthly > 0 ? clamp(intern.monthlyIncome / grossMonthly, 0, 2) : 0,

      dormFactor: dormFactor,
      campusRaw: campusRaw, campusFactor: campusFactor,
      locationFactor: locationFactor,
      prospectRaw: prospectRaw, prospectFactor: prospectFactor,
      numerator: numerator,

      // —— 跨国可比口径（中国等效人民币购买力）——
      pppMul: pppMul,
      pppGrossMonthly: grossMonthly * pppMul,
      pppNetMonthly: netMonthly * pppMul,
      pppBaselineMonthly: baselineMonthly * pppMul,
      pppInternIncome: intern.monthlyIncome * pppMul,
      pppAnnualNet: netMonthly * M * pppMul,

      radar: radarScores({
        dorm: dormFactor, campus: campusFactor, location: locationFactor,
        prospect: prospectFactor, costRatio: costRatio
      })
    };
  }

  function ratingOf(score) {
    for (var i = 0; i < D.RATINGS.length; i++) if (score < D.RATINGS[i].max) return D.RATINGS[i];
    return D.RATINGS[D.RATINGS.length - 1];
  }

  /* ---------------- 雷达五维（0~100，50 = 当地正常水平） ---------------- */
  /* 线性 + clamp 会让维度分顶到 100（实测「前景」经常满分），满分意味着
   * 「再好也没有余地」，既不真实也不好看。改成指数饱和：
   * 中位仍然是 50，越好越趋近 100、越差越趋近 0，但两端都永远够不着。
   * K=1.7 时，本模型定义的「最好情况」落在约 91 分，最差约 9 分。 */
  var SAT_K = 1.7;
  function pivot(v, lo, mid, hi) {
    if (!(hi > mid) || !(mid > lo)) return 50;
    if (v <= mid) {
      var u = Math.max((mid - v) / (mid - lo), 0);        // 0 = 中位，1 = 最差
      return 50 * Math.exp(-SAT_K * u);
    }
    var t = Math.max((v - mid) / (hi - mid), 0);          // 0 = 中位，1 = 最好
    return 100 - 50 * Math.exp(-SAT_K * t);
  }
  function radarScores(r) {
    // 成本是反向的：花得少 = 分高
    var cost = clamp(100 - pivot(r.costRatio, 0.30, 1.0, 2.20), 0, 100);
    return {
      dorm:     pivot(r.dorm,     DORM_RANGE.min,   1.0, DORM_RANGE.max),
      location: pivot(r.location, 0.66,             1.0, 1.24),
      cost:     cost,
      prospect: pivot(r.prospect, PROSPECT_RANGE.min, 1.0, PROSPECT_RANGE.max * 1.35),
      campus:   pivot(r.campus,   CAMPUS_RANGE.min * 0.72, 1.0, CAMPUS_RANGE.max)
    };
  }

  /* =========================================================================
   * 诊断 / 敏感度：把每一项分别拨到最好 / 最差，看得分能动多少
   * ========================================================================= */
  function bestIdx(o) { var b = 0, v = -Infinity; o.forEach(function (x, i) { if (x.v > v) { v = x.v; b = i; } }); return b; }
  function lowIdx(o)  { var b = 0, v = Infinity;  o.forEach(function (x, i) { if (x.v < v) { v = x.v; b = i; } }); return b; }

  // 不可改变的项：填了也没用，不该出现在「值得改善」里
  var IMMUTABLE = { major: 1, schoolTier: 1, localJobs: 1 };

  function buildLevers() {
    var L = [];
    function dimLever(group, d, changeable) {
      L.push({
        group: group, key: d.key, label: d.label, changeable: changeable,
        targetLabel: d.options[bestIdx(d.options)].label,
        to:    function (s) { var n = assign(s, {}); n[d.key] = bestIdx(d.options); return n; },
        worst: function (s) { var n = assign(s, {}); n[d.key] = lowIdx(d.options);  return n; }
      });
    }
    D.DORM_DIMS.forEach(function (d) { dimLever('住宿', d, true); });
    D.CAMPUS_DIMS.forEach(function (d) { dimLever('校园', d, d.key !== 'curfew' && d.key !== 'strictness'); });
    D.PROSPECT_DIMS.forEach(function (d) { dimLever('前景', d, !IMMUTABLE[d.key]); });

    L.push({ group: '地段', key: 'location', label: '校区地段', changeable: false,
      targetLabel: D.LOCATIONS[bestIdx(D.LOCATIONS)].label,
      to:    function (s) { var n = assign(s, {}); n.location = bestIdx(D.LOCATIONS); return n; },
      worst: function (s) { var n = assign(s, {}); n.location = lowIdx(D.LOCATIONS);  return n; } });

    L.push({ group: '地段', key: 'commute', label: '进城交通', changeable: true,
      targetLabel: D.COMMUTE[0].label,
      to:    function (s) { var n = assign(s, {}); n.commute = 0; return n; },
      worst: function (s) { var n = assign(s, {}); n.commute = D.COMMUTE.length - 1; return n; } });

    L.push({ group: '花销', key: 'monthlyLiving', label: '月生活费降 20%', changeable: true,
      to:    function (s) { var n = assign(s, {}); n.monthlyLiving = num(s.monthlyLiving) * 0.8; return n; },
      worst: function (s) { var n = assign(s, {}); n.monthlyLiving = num(s.monthlyLiving) * 1.2; return n; } });

    L.push({ group: '花销', key: 'monthlyHousing', label: '住宿花销降 20%', changeable: true,
      to:    function (s) { var n = assign(s, {}); n.monthlyHousing = num(s.monthlyHousing) * 0.8; return n; },
      worst: function (s) { var n = assign(s, {}); n.monthlyHousing = num(s.monthlyHousing) * 1.2; return n; } });

    L.push({ group: '实习', key: 'internQuality', label: '实习含金量', changeable: true,
      targetLabel: D.INTERN_QUALITY[D.INTERN_QUALITY.length - 1].label,
      to:    function (s) { var n = assign(s, {}); n.internQuality = D.INTERN_QUALITY.length - 1;
                            if (num(s.internDaysPerWeek) <= 0) n.internDaysPerWeek = 3; return n; },
      worst: function (s) { var n = assign(s, {}); n.internQuality = 0; return n; } });

    L.push({ group: '实习', key: 'convertProb', label: '争取到转正机会', changeable: true,
      targetLabel: D.CONVERT_PROB[D.CONVERT_PROB.length - 1].label,
      to:    function (s) { var n = assign(s, {}); n.convertProb = D.CONVERT_PROB.length - 1;
                            if (num(s.internDaysPerWeek) <= 0) { n.internDaysPerWeek = 3; n.internQuality = 2; }
                            return n; },
      worst: function (s) { var n = assign(s, {}); n.convertProb = 0; return n; } });

    L.push({ group: '实习', key: 'internCommute', label: '实习通勤减半', changeable: true,
      to:    function (s) { var n = assign(s, {}); n.internCommute = num(s.internCommute) / 2; return n; },
      worst: function (s) { var n = assign(s, {}); n.internCommute = num(s.internCommute) * 2 + 1; return n; } });

    return L;
  }
  var LEVERS = null;
  function levers() { if (!LEVERS) LEVERS = buildLevers(); return LEVERS; }

  /* 只列真正能动的项 —— 「换个专业」不是建议，是废话 */
  function diagnose(s, topN) {
    var base = compute(s).score;
    var rows = levers().filter(function (lv) { return lv.changeable; }).map(function (lv) {
      var better = compute(lv.to(s)).score;
      return { group: lv.group, key: lv.key, label: lv.label, target: lv.targetLabel || null,
               delta: better - base, pct: base > 0 ? (better - base) / base : 0, to: better };
    }).filter(function (r) { return r.delta > 1e-9; });
    rows.sort(function (a, b) { return b.delta - a.delta; });
    return topN ? rows.slice(0, topN) : rows;
  }

  /* 敏感度：所有项都列，包括不可改变的，用来看「什么决定了这个分数」 */
  function sensitivity(s) {
    var base = compute(s).score;
    var rows = levers().map(function (lv) {
      var hi = compute(lv.to(s)).score, lo = compute(lv.worst(s)).score;
      return { group: lv.group, key: lv.key, label: lv.label, changeable: lv.changeable,
               base: base, hi: hi, lo: lo, up: hi - base, down: lo - base, span: hi - lo };
    });
    rows.sort(function (a, b) { return b.span - a.span; });
    return rows;
  }

  /* ---------------- 格式化 ---------------- */
  function money(v) { return isFinite(v) ? Math.round(v).toLocaleString('en-US') : '—'; }
  function pct(v)   { return (v >= 0 ? '+' : '') + (v * 100).toFixed(1) + '%'; }
  function pct0(v)  { return Math.round(v * 100) + '%'; }

  root.UNI_MODEL = {
    defaultState: defaultState,
    switchCountry: switchCountry,
    compute: compute,
    diagnose: diagnose,
    sensitivity: sensitivity,
    ratingOf: ratingOf,
    weightedFactor: weightedFactor,
    pivot: pivot,
    ranges: { dorm: DORM_RANGE, campus: CAMPUS_RANGE, prospect: PROSPECT_RANGE },
    fmt: { money: money, pct: pct, pct0: pct0 },
    clamp: clamp, assign: assign
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = root.UNI_MODEL;

})(typeof window !== 'undefined' ? window : globalThis);
