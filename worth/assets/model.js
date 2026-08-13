/* ===========================================================================
 * model.js —— 计算引擎（纯函数，无 DOM 依赖，可在 Node 里直接跑）
 * ---------------------------------------------------------------------------
 *                    标准化日薪 × 环境系数 × 成长系数
 *   工作性价比  =  ────────────────────────────────────────────────
 *                  基准时薪 × 有效工时 × 期望系数 × 风险系数
 *
 *   得分 1.00 = 在**所在国**刚好对得起你的年限、学历与时间投入
 *   跨国对比另有 PPP 标准化指标（pppAdjHourly / pppTC），见下方说明
 * =========================================================================== */

(function (root) {
  'use strict';

  var D = root.PJC_DATA || (typeof require !== 'undefined' ? require('./data.js') : null);
  var G = D.GLOBAL;

  /* ---------------- 小工具 ---------------- */
  function clamp(x, lo, hi) { return Math.min(Math.max(x, lo), hi); }
  function num(x, d) { var v = parseFloat(x); return isFinite(v) ? v : (d || 0); }
  function sum(a) { return a.reduce(function (x, y) { return x + y; }, 0); }
  function assign(a, b) { var o = {}, k; for (k in a) o[k] = a[k]; for (k in b) o[k] = b[k]; return o; }

  function country(cc) { return D.COUNTRIES[cc] || D.COUNTRIES.CN; }

  /* ---------------- 默认状态 ---------------- */
  function defaultState(cc) {
    cc = cc || 'CN';
    var s = { country: cc };
    D.buildSections(cc).forEach(function (sec) {
      sec.fields.forEach(function (f) {
        s[f.key] = Array.isArray(f.def) ? f.def.slice() : f.def;
      });
    });
    return s;
  }

  /* 换国家：保留跨国通用的填写，重置国家专属项 */
  var COUNTRY_SPECIFIC = [
    'monthlyBase', 'grossAnnual', 'salaryMonths', 'bonusCash', 'allowanceMonthly',
    'stockAnnual', 'fundRate', 'includeCompanyFund', 'specialDeduct', 'socialCap',
    'workDaysPerWeek', 'dailyHours', 'commuteHours', 'annualLeave', 'publicHolidays',
    'sickLeave', 'leaveHard', 'city', 'companyType', 'contract',
    'degree', 'school', 'bachelorSchool', 'workYears', 'age'
  ];
  function switchCountry(state, cc) {
    var fresh = defaultState(cc);
    var out = assign(fresh, {});
    Object.keys(state).forEach(function (k) {
      if (k === 'country') return;
      if (COUNTRY_SPECIFIC.indexOf(k) >= 0) return;   // 用新国家的默认值
      if (fresh.hasOwnProperty(k)) out[k] = state[k];  // 通用项沿用
    });
    // 年限、学历、年龄这些「人不会因为换国家而改变」的，尽量沿用索引
    ['workYears', 'degree', 'school', 'bachelorSchool', 'age'].forEach(function (k) {
      if (state[k] !== undefined && fresh.hasOwnProperty(k)) out[k] = state[k];
    });
    out.country = cc;
    return out;
  }

  /* ---------------- 税与社保 ---------------- */
  function quickTax(taxable, brackets) {
    if (taxable <= 0) return 0;
    for (var i = 0; i < brackets.length; i++) {
      if (taxable <= brackets[i][0]) return taxable * brackets[i][1] - brackets[i][2];
    }
    return 0;
  }
  function marginalTax(base, brackets) {
    if (base <= 0) return 0;
    var acc = 0, prev = 0;
    for (var i = 0; i < brackets.length; i++) {
      var cap = brackets[i][0], rate = brackets[i][1];
      if (base <= cap) { acc += (base - prev) * rate; return acc; }
      acc += (cap - prev) * rate; prev = cap;
    }
    return acc;
  }

  /* 返回 { personalSocial, personalFund, companyFund, tax, net } */
  function payroll(C, s, cashAnnual, stockValue) {
    var socialCapM = num(s.socialCap, C.social.capMonthly);
    var taxableCash = cashAnnual + stockValue;

    if (C.key === 'CN') {
      var fundRate = C.social.fundRates[clamp(s.fundRate | 0, 0, C.social.fundRates.length - 1)];
      var fundBase = Math.min(num(s.monthlyBase), socialCapM);
      var companyFund = s.includeCompanyFund ? fundBase * fundRate * 12 : 0;
      var personalSocial = fundBase * C.social.rate * 12;
      var personalFund = fundBase * fundRate * 12;
      var deductible = C.tax.basicDeduction + personalSocial + personalFund + num(s.specialDeduct) * 12;
      var tax = quickTax(Math.max(taxableCash - deductible, 0), C.tax.brackets);
      // 个人公积金进的是自己账户，不算流失
      return {
        personalSocial: personalSocial, personalFund: personalFund, companyFund: companyFund,
        tax: tax, net: taxableCash - personalSocial - tax + companyFund
      };
    }

    // 西班牙：Seguridad Social（有 base máxima）+ IRPF
    var ssBase = Math.min(taxableCash / 12, socialCapM) * 12;
    var ss = ssBase * C.social.rate;
    var base = Math.max(taxableCash - ss - C.tax.gastosDeducibles - num(s.specialDeduct), 0);
    var cuota = marginalTax(base, C.tax.brackets) -
                marginalTax(Math.min(C.tax.minimoPersonal, base), C.tax.brackets);
    cuota = Math.max(cuota, 0);
    return {
      personalSocial: ss, personalFund: 0, companyFund: 0,
      tax: cuota, net: taxableCash - ss - cuota
    };
  }

  /* ---------------- 加权平均 ---------------- */
  function weightedFactor(dims, s) {
    var tw = 0, acc = 0;
    dims.forEach(function (d) {
      var i = clamp(num(s[d.key], 0) | 0, 0, d.options.length - 1);
      acc += d.weight * d.options[i].v; tw += d.weight;
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
  var GROWTH_RANGE = weightedRange(D.GROWTH_DIMS);
  var ENV_RANGE = weightedRange(D.ENV_DIMS);
  var PERK_MAX = sum(D.PERKS.map(function (p) { return p.v; }));

  /* ---------------- 学历系数 ---------------- */
  function educationFactor(C, s) {
    var T = C.eduTable;
    var deg = C.degrees[clamp(s.degree | 0, 0, C.degrees.length - 1)].key;
    var sch = C.schools[clamp(s.school | 0, 0, C.schools.length - 1)].key;
    if (deg === 'below') return T.below.fixed;
    if (deg === 'bachelor') return T.bachelor[sch];
    if (deg === 'master') {
      var bs = C.schools[clamp(s.bachelorSchool | 0, 0, C.schools.length - 1)].key;
      return T.masterBase[bs] + T.masterBonus[sch];
    }
    return T.phd[sch];
  }

  function ageBandOf(C, age) {
    for (var i = 0; i < C.ageRisk.length; i++) if (age < C.ageRisk[i].max) return C.ageRisk[i];
    return C.ageRisk[C.ageRisk.length - 1];
  }

  /* =========================================================================
   * 主计算
   * ========================================================================= */
  function compute(s) {
    var C = country(s.country);

    /* 1. 年实际工作天数 */
    var wdOpt = C.weekDayOptions[clamp(s.workDaysPerWeek | 0, 0, C.weekDayOptions.length - 1)];
    var wdpw = wdOpt.v;
    var leaveHardV = D.LEAVE_HARD[clamp(s.leaveHard | 0, 0, D.LEAVE_HARD.length - 1)].v;
    var usableLeave = num(s.annualLeave) * leaveHardV;
    var totalLeave = usableLeave + num(s.publicHolidays) + num(s.sickLeave) * G.SICK_LEAVE_WEIGHT;
    var workDays = Math.max(G.WEEKS_PER_YEAR * wdpw - totalLeave, 1);

    /* 2. 年总包 */
    var baseCash = C.salaryMode === 'monthly'
      ? num(s.monthlyBase) * num(s.salaryMonths)
      : num(s.grossAnnual);
    var cashAnnual = baseCash + num(s.bonusCash) + num(s.allowanceMonthly) * 12;
    var stockRaw = num(s.stockAnnual);
    var stockDiscount = D.STOCK_DISCOUNT[clamp(s.stockType | 0, 0, D.STOCK_DISCOUNT.length - 1)].v;
    var stockValue = stockRaw * stockDiscount;

    var pr = payroll(C, s, cashAnnual, stockValue);
    var tcGross = cashAnnual + stockValue + pr.companyFund;
    var tcNet = pr.net;
    var tc = s.afterTax ? tcNet : tcGross;
    var dailySalary = tc / workDays;

    /* 3. 有效工时 */
    var dailyHours = num(s.dailyHours);
    var baseH = Math.min(dailyHours, G.STD_DAY_HOURS);
    var otComp = D.OVERTIME_COMP[clamp(s.overtimeComp | 0, 0, D.OVERTIME_COMP.length - 1)].v;
    var otH = Math.max(dailyHours - G.STD_DAY_HOURS, 0) * otComp;
    var wfh = Math.min(num(s.wfhDays), wdpw);
    var officeRatio = wdpw > 0 ? (wdpw - wfh) / wdpw : 0;
    var comfort = D.COMMUTE_COMFORT[clamp(s.commuteComfort | 0, 0, D.COMMUTE_COMFORT.length - 1)].v;
    var commuteH = num(s.commuteHours) * officeRatio * comfort;
    var oncallH = D.ONCALL[clamp(s.oncall | 0, 0, D.ONCALL.length - 1)].v;
    var slackH = G.SLACK_WEIGHT * num(s.slackHours);
    var effHoursRaw = baseH + otH + commuteH + oncallH - slackH;
    var effHours = Math.max(effHoursRaw, G.MIN_EFF_HOURS);

    /* 4~5. 成长与环境 */
    var growthFactor = weightedFactor(D.GROWTH_DIMS, s);
    var envHuman = weightedFactor(D.ENV_DIMS, s);
    var perkKeys = Array.isArray(s.perks) ? s.perks : [];
    var perkBonus = sum(D.PERKS.filter(function (p) { return perkKeys.indexOf(p.key) >= 0; })
                              .map(function (p) { return p.v; }));
    var cityFactor = C.cities[clamp(s.city | 0, 0, C.cities.length - 1)].v;
    var envFactor = cityFactor * (envHuman + perkBonus);

    /* 6. 期望系数 */
    var comp = C.companyTypes[clamp(s.companyType | 0, 0, C.companyTypes.length - 1)];
    var eduFactor = educationFactor(C, s);
    var yearBase = C.years[clamp(s.workYears | 0, 0, C.years.length - 1)].v;
    var yearFactor = 1 + (yearBase - 1) * comp.track;
    var expectFactor = eduFactor * yearFactor;

    /* 7. 风险系数 */
    var riskAdds = D.RISK_DIMS.map(function (d) {
      return d.options[clamp(num(s[d.key], 0) | 0, 0, d.options.length - 1)].v;
    });
    var contractAdd = C.contractDim.options[clamp(num(s.contract, 0) | 0, 0, C.contractDim.options.length - 1)].v;
    var ageBand = ageBandOf(C, num(s.age, 30));
    var ageAdd = ageBand.v * comp.market;
    var riskFactorRaw = comp.risk + sum(riskAdds) + contractAdd + ageAdd;
    var riskFactor = clamp(riskFactorRaw, 0.70, 2.40);

    /* 8. 得分
     * 基准时薪按税前中位数标定；切到税后口径时同步下调，否则分子税后、
     * 分母税前，会把所有人系统性压分（且高税国家被压得更狠）。 */
    var baseHourly = C.baseHourly * (s.afterTax ? (C.taxRatio || 1) : 1);
    var numerator = dailySalary * envFactor * growthFactor;
    var denominator = baseHourly * effHours * expectFactor * riskFactor;
    var score = denominator > 0 ? numerator / denominator : 0;

    /* 9. 衍生指标 */
    var rawHourly = dailySalary / effHours;
    var adjHourly = rawHourly * envFactor * growthFactor;
    var deservedHourly = baseHourly * expectFactor * riskFactor;

    // PPP 标准化：折算成「中国等效人民币购买力」，用于跨国横向比较
    var pppMul = G.PPP_ANCHOR / C.ppp;

    return {
      country: C, cur: C.cur, score: score, rating: ratingOf(score),

      workDays: workDays, wdpw: wdpw,
      baseCash: baseCash, cashAnnual: cashAnnual,
      stockRaw: stockRaw, stockValue: stockValue, stockDiscount: stockDiscount,
      companyFund: pr.companyFund, personalSocial: pr.personalSocial,
      personalFund: pr.personalFund, tax: pr.tax,
      tcGross: tcGross, tcNet: tcNet, tc: tc,
      effectiveTaxRate: (cashAnnual + stockValue) > 0
        ? (pr.personalSocial + pr.tax) / (cashAnnual + stockValue) : 0,
      dailySalary: dailySalary,

      effHours: effHours, effHoursRaw: effHoursRaw,
      breakdown: { baseH: baseH, otH: otH, commuteH: commuteH, oncallH: oncallH,
                   slackH: slackH, officeRatio: officeRatio },

      growthFactor: growthFactor, envHuman: envHuman, perkBonus: perkBonus,
      cityFactor: cityFactor, envFactor: envFactor,
      eduFactor: eduFactor, yearFactor: yearFactor, expectFactor: expectFactor,
      riskFactor: riskFactor, riskFactorRaw: riskFactorRaw, riskAdds: riskAdds,
      contractAdd: contractAdd, ageAdd: ageAdd, ageBand: ageBand, company: comp,

      baseHourly: baseHourly,
      rawHourly: rawHourly, adjHourly: adjHourly, deservedHourly: deservedHourly,
      annualHours: workDays * effHours,

      // —— 跨国可比口径（中国等效人民币购买力）——
      pppMul: pppMul,
      pppTC: tc * pppMul,
      pppRawHourly: rawHourly * pppMul,
      pppAdjHourly: adjHourly * pppMul,
      pppDeservedHourly: deservedHourly * pppMul,

      numerator: numerator, denominator: denominator,
      radar: radarScores({
        dailySalary: dailySalary, effHours: effHours, baseHourly: baseHourly,
        expectFactor: expectFactor, growthFactor: growthFactor,
        envHuman: envHuman, perkBonus: perkBonus, riskFactor: riskFactor
      })
    };
  }

  function ratingOf(score) {
    for (var i = 0; i < D.RATINGS.length; i++) if (score < D.RATINGS[i].max) return D.RATINGS[i];
    return D.RATINGS[D.RATINGS.length - 1];
  }

  /* ---------------- 雷达五维（0~100，50 = 一切正常） ---------------- */
  function pivot(v, lo, mid, hi) {
    if (v <= mid) return clamp((v - lo) / (mid - lo) * 50, 0, 50);
    return clamp(50 + (v - mid) / (hi - mid) * 50, 50, 100);
  }
  var NEUTRAL_HOURS = 9.5;

  function radarScores(r) {
    var payRatio = r.dailySalary / (r.baseHourly * r.effHours * r.expectFactor);
    var pay = payRatio <= 0 ? 0 : clamp(50 * (1 + Math.log(payRatio) / Math.log(3)), 0, 100);
    var time = clamp(50 + (NEUTRAL_HOURS - r.effHours) * 10, 0, 100);
    var growth = pivot(r.growthFactor, GROWTH_RANGE.min, 1.0, GROWTH_RANGE.max);
    var env = pivot(r.envHuman + r.perkBonus, ENV_RANGE.min, 1.0, ENV_RANGE.max + PERK_MAX);
    var stable = clamp(100 - pivot(r.riskFactor, 0.82, 1.0, 1.80), 0, 100);
    return { pay: pay, time: time, growth: growth, env: env, stable: stable };
  }

  /* =========================================================================
   * 薪资求解器
   * -------------------------------------------------------------------------
   * 给定一份「除薪资外全部固定」的处境，求薪资要多少才能达到 targetScore。
   * 得分对薪资单调递增（税后模式下因累进税而非线性），用二分法最稳。
   * ========================================================================= */
  function salaryFieldOf(cc) {
    return country(cc).salaryMode === 'monthly' ? 'monthlyBase' : 'grossAnnual';
  }

  /* 把薪资字段设成 v，并按原比例缩放 bonus / 股票（保持薪酬结构不变） */
  function withSalary(s, v) {
    var f = salaryFieldOf(s.country);
    var cur = num(s[f], 0);
    var out = assign(s, {});
    out[f] = v;
    if (cur > 0) {
      var k = v / cur;
      out.bonusCash = num(s.bonusCash) * k;
      out.stockAnnual = num(s.stockAnnual) * k;
      out.allowanceMonthly = num(s.allowanceMonthly) * k;
    }
    return out;
  }

  function solveSalary(s, targetScore, opts) {
    opts = opts || {};
    var keepStructure = opts.keepStructure !== false;
    var f = salaryFieldOf(s.country);
    var set = keepStructure ? withSalary : function (st, v) { var o = assign(st, {}); o[f] = v; return o; };

    if (!(targetScore > 0)) return null;

    var lo = 0, hi = Math.max(num(s[f], 1) * 4, country(s.country).salaryMode === 'monthly' ? 20000 : 60000);
    // 先把上界撑到足够大
    var guard = 0;
    while (compute(set(s, hi)).score < targetScore && guard++ < 40) hi *= 2;
    if (guard >= 40) return null;

    for (var i = 0; i < 80; i++) {
      var mid = (lo + hi) / 2;
      if (compute(set(s, mid)).score < targetScore) lo = mid; else hi = mid;
    }
    var value = (lo + hi) / 2;
    var result = compute(set(s, value));
    return {
      field: f,
      value: value,
      annual: country(s.country).salaryMode === 'monthly'
        ? value * num(s.salaryMonths, 12) : value,
      state: set(s, value),
      result: result
    };
  }

  /* 差异归因：把 scoreA / scoreB 精确拆成七个乘法因子
   *
   *   score = 日薪 × 环境 × 成长 ÷ (基准时薪 × 有效工时 × 期望 × 风险)
   *
   * 分子分母同乘 PPP 系数不改变得分，所以用 PPP 标准化后的日薪与基准时薪，
   * 就能把两个国家放在同一把尺子上比。七个因子连乘 = scoreA / scoreB。 */
  function attribution(rA, rB) {
    var dsA = rA.dailySalary * rA.pppMul, dsB = rB.dailySalary * rB.pppMul;
    var bhA = rA.baseHourly * rA.pppMul, bhB = rB.baseHourly * rB.pppMul;
    var rows = [
      { key: 'pay',    label: '实际到手（PPP 折算后的日薪）', v: dsB > 0 ? dsA / dsB : 1,
        a: dsA, b: dsB, unit: '¥/天' },
      { key: 'hours',  label: '有效工时（越短越有利）',        v: rA.effHours > 0 ? rB.effHours / rA.effHours : 1,
        a: rA.effHours, b: rB.effHours, unit: 'h/天', inverse: true },
      { key: 'market', label: '本国市场基准时薪',              v: bhA > 0 ? bhB / bhA : 1,
        a: bhA, b: bhB, unit: '¥/h', inverse: true },
      { key: 'expect', label: '年限 × 学历的薪资期望',         v: rA.expectFactor > 0 ? rB.expectFactor / rA.expectFactor : 1,
        a: rA.expectFactor, b: rB.expectFactor, unit: '', inverse: true },
      { key: 'env',    label: '城市成本 × 团队环境',           v: rB.envFactor > 0 ? rA.envFactor / rB.envFactor : 1,
        a: rA.envFactor, b: rB.envFactor, unit: '' },
      { key: 'growth', label: '技术成长',                     v: rB.growthFactor > 0 ? rA.growthFactor / rB.growthFactor : 1,
        a: rA.growthFactor, b: rB.growthFactor, unit: '' },
      { key: 'risk',   label: '风险',                         v: rA.riskFactor > 0 ? rB.riskFactor / rA.riskFactor : 1,
        a: rA.riskFactor, b: rB.riskFactor, unit: '', inverse: true }
    ];
    var product = rows.reduce(function (p, r) { return p * r.v; }, 1);
    return { rows: rows, product: product, ratio: rB.score > 0 ? rA.score / rB.score : 0 };
  }

  /* 跨国对照：A 国的处境 vs B 国的处境，以及互相追平所需的薪资 */
  function crossCompare(stateA, stateB) {
    var rA = compute(stateA), rB = compute(stateB);
    return {
      a: rA, b: rB,
      attribution: attribution(rA, rB),
      // B 国要拿多少才能追平 A 国的得分
      bNeedsToMatchA: solveSalary(stateB, rA.score),
      // A 国要拿多少才能追平 B 国的得分
      aNeedsToMatchB: solveSalary(stateA, rB.score),
      // 纯购买力等值（只换算钱，不考虑工时/环境/成长/风险）
      pppOnly: {
        aTCinB: rA.tc * (rA.pppMul / rB.pppMul),
        bTCinA: rB.tc * (rB.pppMul / rA.pppMul)
      }
    };
  }

  /* =========================================================================
   * 杠杆 / 诊断 / 敏感度
   * ========================================================================= */
  function bestIdx(o) { var b = 0, v = -Infinity; o.forEach(function (x, i) { if (x.v > v) { v = x.v; b = i; } }); return b; }
  function lowIdx(o) { var b = 0, v = Infinity; o.forEach(function (x, i) { if (x.v < v) { v = x.v; b = i; } }); return b; }

  function buildLevers(cc) {
    var C = country(cc);
    var L = [];

    D.GROWTH_DIMS.forEach(function (d) {
      L.push({ group: '成长', key: d.key, label: d.label,
        to: function (s) { var n = assign(s, {}); n[d.key] = bestIdx(d.options); return n; },
        worst: function (s) { var n = assign(s, {}); n[d.key] = lowIdx(d.options); return n; },
        targetLabel: d.options[bestIdx(d.options)].label });
    });
    D.ENV_DIMS.forEach(function (d) {
      L.push({ group: '环境', key: d.key, label: d.label,
        to: function (s) { var n = assign(s, {}); n[d.key] = bestIdx(d.options); return n; },
        worst: function (s) { var n = assign(s, {}); n[d.key] = lowIdx(d.options); return n; },
        targetLabel: d.options[bestIdx(d.options)].label });
    });
    D.RISK_DIMS.concat([C.contractDim]).forEach(function (d) {
      L.push({ group: '风险', key: d.key, label: d.label,
        to: function (s) { var n = assign(s, {}); n[d.key] = lowIdx(d.options); return n; },
        worst: function (s) { var n = assign(s, {}); n[d.key] = bestIdx(d.options); return n; },
        targetLabel: d.options[lowIdx(d.options)].label });
    });

    L.push({ group: '时间', key: 'dailyHours', label: '日均在司时长 −2 小时',
      to: function (s) { var n = assign(s, {}); n.dailyHours = Math.max(num(s.dailyHours) - 2, 4); return n; },
      worst: function (s) { var n = assign(s, {}); n.dailyHours = num(s.dailyHours) + 2; return n; } });
    L.push({ group: '时间', key: 'commuteHours', label: '通勤时长减半',
      to: function (s) { var n = assign(s, {}); n.commuteHours = num(s.commuteHours) / 2; return n; },
      worst: function (s) { var n = assign(s, {}); n.commuteHours = num(s.commuteHours) * 2; return n; } });
    L.push({ group: '时间', key: 'wfhDays', label: '每周多 2 天远程',
      to: function (s) { var n = assign(s, {}); n.wfhDays = Math.min(num(s.wfhDays) + 2, C.weekDayOptions[s.workDaysPerWeek | 0].v); return n; },
      worst: function (s) { var n = assign(s, {}); n.wfhDays = 0; return n; } });
    L.push({ group: '时间', key: 'oncall', label: 'On-call 强度',
      to: function (s) { var n = assign(s, {}); n.oncall = 0; return n; },
      worst: function (s) { var n = assign(s, {}); n.oncall = D.ONCALL.length - 1; return n; },
      targetLabel: '取消值班' });
    L.push({ group: '时间', key: 'overtimeComp', label: '加班补偿',
      to: function (s) { var n = assign(s, {}); n.overtimeComp = D.OVERTIME_COMP.length - 1; return n; },
      worst: function (s) { var n = assign(s, {}); n.overtimeComp = 0; return n; },
      targetLabel: D.OVERTIME_COMP[D.OVERTIME_COMP.length - 1].label });
    L.push({ group: '时间', key: 'commuteComfort', label: '通勤方式',
      to: function (s) { var n = assign(s, {}); n.commuteComfort = 0; return n; },
      worst: function (s) { var n = assign(s, {}); n.commuteComfort = 3; return n; },
      targetLabel: D.COMMUTE_COMFORT[0].label });
    L.push({ group: '时间', key: 'leaveHard', label: '年假可用性',
      to: function (s) { var n = assign(s, {}); n.leaveHard = 0; return n; },
      worst: function (s) { var n = assign(s, {}); n.leaveHard = D.LEAVE_HARD.length - 1; return n; },
      targetLabel: D.LEAVE_HARD[0].label });

    L.push({ group: '薪酬', key: 'salary', label: '总包涨 20%',
      to: function (s) { return withSalary(s, num(s[salaryFieldOf(s.country)]) * 1.2); },
      worst: function (s) { return withSalary(s, num(s[salaryFieldOf(s.country)]) * 0.8); } });

    // 城市：敏感度跑满量程，诊断只建议现实的一档
    var midCity = Math.min(4, C.cities.length - 1);
    L.push({ group: '环境', key: 'city', label: '城市生活成本',
      to: function (s) { var n = assign(s, {}); n.city = bestIdx(C.cities); return n; },
      diagTo: function (s) { var n = assign(s, {}); n.city = midCity; return n; },
      worst: function (s) { var n = assign(s, {}); n.city = lowIdx(C.cities); return n; },
      diagTargetLabel: C.cities[midCity].label.split('（')[0],
      targetLabel: C.cities[bestIdx(C.cities)].label });

    return L;
  }

  var LEVER_CACHE = {};
  function leversFor(cc) {
    if (!LEVER_CACHE[cc]) LEVER_CACHE[cc] = buildLevers(cc);
    return LEVER_CACHE[cc];
  }

  function diagnose(s, topN) {
    var base = compute(s).score;
    var rows = leversFor(s.country).map(function (lv) {
      var apply = lv.diagTo || lv.to;
      var better = compute(apply(s)).score;
      return { group: lv.group, key: lv.key, label: lv.label,
               target: lv.diagTargetLabel || lv.targetLabel || null,
               delta: better - base, pct: base > 0 ? (better - base) / base : 0, to: better };
    }).filter(function (r) { return r.delta > 1e-9; });
    rows.sort(function (a, b) { return b.delta - a.delta; });
    return topN ? rows.slice(0, topN) : rows;
  }

  function sensitivity(s) {
    var base = compute(s).score;
    var rows = leversFor(s.country).map(function (lv) {
      var hi = compute(lv.to(s)).score, lo = compute(lv.worst(s)).score;
      return { group: lv.group, key: lv.key, label: lv.label,
               base: base, hi: hi, lo: lo, up: hi - base, down: lo - base, span: hi - lo };
    });
    rows.sort(function (a, b) { return b.span - a.span; });
    return rows;
  }

  /* ---------------- 格式化 ---------------- */
  function money(v) { return isFinite(v) ? Math.round(v).toLocaleString('en-US') : '—'; }
  function wan(v, cc) {
    if (!isFinite(v)) return '—';
    if (cc === 'ES') return (v / 1000).toFixed(1) + 'k';
    return (v / 10000).toFixed(1) + ' 万';
  }
  function pct(v) { return (v >= 0 ? '+' : '') + (v * 100).toFixed(1) + '%'; }

  root.PJC_MODEL = {
    defaultState: defaultState,
    switchCountry: switchCountry,
    compute: compute,
    diagnose: diagnose,
    sensitivity: sensitivity,
    ratingOf: ratingOf,
    solveSalary: solveSalary,
    salaryFieldOf: salaryFieldOf,
    withSalary: withSalary,
    crossCompare: crossCompare,
    attribution: attribution,
    educationFactor: educationFactor,
    payroll: payroll,
    ranges: { growth: GROWTH_RANGE, env: ENV_RANGE, perkMax: PERK_MAX },
    fmt: { money: money, wan: wan, pct: pct },
    clamp: clamp, assign: assign
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = root.PJC_MODEL;

})(typeof window !== 'undefined' ? window : globalThis);
