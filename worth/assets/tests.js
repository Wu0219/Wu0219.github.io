/* ===========================================================================
 * tests.js —— 模型标定用例（国际版）
 * ---------------------------------------------------------------------------
 * 浏览器：打开 tests.html      命令行：node assets/tests.js
 *
 * 这些不是「单元测试」而是**标定测试**：把两国市场上真实可识别的处境代进模型，
 * 检查得分是否落在符合直觉的区间。改动任何系数后重跑，就知道刻度有没有调歪。
 * =========================================================================== */

(function (root) {
  'use strict';

  var D = root.PJC_DATA || require('./data.js');
  var M = root.PJC_MODEL || require('./model.js');

  function co(cc, key) {
    var l = D.COUNTRIES[cc].companyTypes;
    for (var i = 0; i < l.length; i++) if (l[i].key === key) return i;
    return 0;
  }
  function op(dims, dimKey, needle) {
    for (var i = 0; i < dims.length; i++) if (dims[i].key === dimKey) {
      for (var j = 0; j < dims[i].options.length; j++)
        if (dims[i].options[j].label.indexOf(needle) >= 0) return j;
    }
    return 0;
  }
  function mk(cc, p) { return M.assign(M.defaultState(cc), p); }

  /* ------------------------------------------------------------------
   * 标定用例
   * ------------------------------------------------------------------ */
  var CASES = [
    /* ---------- 🇨🇳 中国 ---------- */
    {
      cc: 'CN', name: '北京一线大厂 · P6 · 3~5 年 · 双非本科 · 总包 68 万',
      note: '对标 2025 阿里 P6 / 字节 2-1 区间，应落在中位偏上。',
      expect: [0.95, 1.55],
      state: mk('CN', { monthlyBase: 35000, salaryMonths: 16, stockAnnual: 80000, stockType: 1,
        fundRate: 4, city: 0, companyType: co('CN', 'bigtech'), dailyHours: 10.5, commuteHours: 1.5,
        commuteComfort: 3, slackHours: 1.5, oncall: 1, annualLeave: 10, leaveHard: 1,
        workYears: 2, age: 28, degree: 1, school: 1 })
    },
    {
      cc: 'CN', name: '二线国企 · 5~8 年 · 月薪 15k × 14 · 955',
      note: '钱少事少。期望与风险都被大幅下调，应在中位附近。',
      expect: [0.70, 1.30],
      state: mk('CN', { monthlyBase: 15000, salaryMonths: 14, fundRate: 4, city: 3,
        companyType: co('CN', 'soe'), dailyHours: 8.5, commuteHours: 1, slackHours: 2,
        annualLeave: 10, leaveHard: 0, workYears: 3, age: 31,
        techStack: op(D.GROWTH_DIMS, 'techStack', '传统企业开发'),
        codebase: op(D.GROWTH_DIMS, 'codebase', '屎山'),
        autonomy: op(D.GROWTH_DIMS, 'autonomy', '改配置') })
    },
    {
      cc: 'CN', name: '外包驻场 · 996 · 3~5 年 · 月薪 18k × 12',
      note: '钱不多、成长为零、风险最高。模型应给出明确的「跑」。',
      expect: [0.0, 0.50],
      state: mk('CN', { monthlyBase: 18000, salaryMonths: 12, fundRate: 0, city: 3,
        companyType: co('CN', 'outsource'), workDaysPerWeek: 4, dailyHours: 12, commuteHours: 1.5,
        commuteComfort: 3, slackHours: 0.5, annualLeave: 5, leaveHard: 2, workYears: 2, age: 29, contract: 2,
        techStack: op(D.GROWTH_DIMS, 'techStack', '传统企业开发'),
        autonomy: op(D.GROWTH_DIMS, 'autonomy', '外派驻场'),
        codebase: op(D.GROWTH_DIMS, 'codebase', '考古现场'),
        engineering: op(D.GROWTH_DIMS, 'engineering', '全靠手动'),
        leader: op(D.ENV_DIMS, 'leader', '外行指挥'),
        office: op(D.ENV_DIMS, 'office', '无固定工位') })
    },
    {
      cc: 'CN', name: '应届生 · 一线大厂 · 总包 30 万',
      note: '应届期望系数为 1，30 万在北京算不错的起点。',
      expect: [1.10, 1.90],
      state: mk('CN', { monthlyBase: 20000, salaryMonths: 15, fundRate: 4, city: 0,
        companyType: co('CN', 'bigtech'), dailyHours: 10.5, commuteHours: 1.5, commuteComfort: 3,
        slackHours: 1.5, oncall: 1, annualLeave: 10, leaveHard: 1, workYears: 0, age: 23 })
    },
    {
      cc: 'CN', name: '40 岁 · 中小厂 · 8~10 年 · 45 万 · 业务边缘 + 传闻裁员',
      note: '年龄风险 + 业务风险叠加，应明显偏低。',
      expect: [0.30, 0.95],
      state: mk('CN', { monthlyBase: 30000, salaryMonths: 14, fundRate: 2, city: 2,
        companyType: co('CN', 'sme'), dailyHours: 10, commuteHours: 1.5, slackHours: 1.5, oncall: 1,
        annualLeave: 10, leaveHard: 1, workYears: 4, age: 40, bizHealth: 2, layoff: 1, salaryInvert: 2 })
    },

    /* ---------- 🇪🇸 西班牙 ---------- */
    {
      cc: 'ES', name: 'Junior · Madrid · producto · €26k',
      note: '马德里 junior 起薪的典型值，应贴近 1.00。',
      expect: [0.85, 1.20],
      state: mk('ES', { grossAnnual: 26000, city: 0, companyType: co('ES', 'product'),
        dailyHours: 8.5, commuteHours: 1, slackHours: 1.5, annualLeave: 23, leaveHard: 0,
        workYears: 0, age: 24 })
    },
    {
      cc: 'ES', name: 'Senior · Madrid · producto · €55k · 5~8 年',
      note: 'senior 6~7 年常超 50k，€55k 属于市场中位。',
      expect: [0.85, 1.30],
      state: mk('ES', { grossAnnual: 55000, city: 0, companyType: co('ES', 'product'),
        dailyHours: 8.5, commuteHours: 1, slackHours: 1.5, annualLeave: 23, leaveHard: 0,
        workYears: 3, age: 33 })
    },
    {
      cc: 'ES', name: 'Cárnica / 外派 · Madrid · senior · €38k',
      note: '马德里中型咨询 senior 常只有 38~42k 且无 equity；成长与风险双杀。',
      expect: [0.0, 0.75],
      state: mk('ES', { grossAnnual: 38000, city: 0, companyType: co('ES', 'carnica'),
        dailyHours: 9, commuteHours: 1.25, slackHours: 1, annualLeave: 22, leaveHard: 1,
        workYears: 3, age: 33, contract: 2,
        techStack: op(D.GROWTH_DIMS, 'techStack', '传统企业开发'),
        autonomy: op(D.GROWTH_DIMS, 'autonomy', '外派驻场'),
        codebase: op(D.GROWTH_DIMS, 'codebase', '屎山'),
        engineering: op(D.GROWTH_DIMS, 'engineering', '全靠手动'),
        office: op(D.ENV_DIMS, 'office', '无固定工位') })
    },
    {
      cc: 'ES', name: 'Big Tech · Madrid · €90k + €25k RSU · 5~8 年',
      note: '西班牙薪资天花板区间，应明显高于中位。',
      expect: [1.55, 2.60],
      state: mk('ES', { grossAnnual: 90000, stockAnnual: 25000, stockType: 1, city: 0,
        companyType: co('ES', 'bigtech'), dailyHours: 8.5, commuteHours: 1, wfhDays: 2,
        slackHours: 1.5, annualLeave: 25, leaveHard: 0, workYears: 3, age: 33, oncall: 1 })
    },
    {
      cc: 'ES', name: '银行 IT · Madrid · €45k · 8~10 年 · 955',
      note: '稳定、955、convenio 好，但技术栈老、涨薪期望低。应在中位附近。',
      expect: [0.75, 1.25],
      state: mk('ES', { grossAnnual: 45000, city: 0, companyType: co('ES', 'banca'),
        dailyHours: 8, commuteHours: 1, wfhDays: 2, slackHours: 2, annualLeave: 25, leaveHard: 0,
        workYears: 4, age: 38,
        techStack: op(D.GROWTH_DIMS, 'techStack', '传统企业开发'),
        codebase: op(D.GROWTH_DIMS, 'codebase', '屎山') })
    },
    {
      cc: 'ES', name: '100% 远程 · 住低成本地区 · €50k · 5~8 年',
      note: '拿大城市薪水、付小地方房租、零通勤 —— 应该是明显划算的组合。',
      expect: [1.10, 1.90],
      state: mk('ES', { grossAnnual: 50000, city: 7, companyType: co('ES', 'product'),
        dailyHours: 8, commuteHours: 0, wfhDays: 5, slackHours: 1.5, annualLeave: 24,
        leaveHard: 0, workYears: 3, age: 32 })
    }
  ];

  /* ------------------------------------------------------------------
   * 税后口径校验（对照公开工资计算器）
   * ------------------------------------------------------------------ */
  var NET_CHECKS = [
    { cc: 'ES', gross: 25000, expectNet: [19300, 20400], label: '🇪🇸 bruto €25.000 → neto 年' },
    { cc: 'ES', gross: 40000, expectNet: [29000, 30300], label: '🇪🇸 bruto €40.000 → neto 年' },
    { cc: 'ES', gross: 70000, expectNet: [46500, 48500], label: '🇪🇸 bruto €70.000 → neto 年' }
  ];

  /* ------------------------------------------------------------------
   * 不变量
   * ------------------------------------------------------------------ */
  var INVARIANTS = [
    { name: '两国：得分对薪资单调递增', run: function () {
      return ['CN', 'ES'].every(function (c) {
        var s = M.defaultState(c);
        var f = M.salaryFieldOf(c);
        var a = M.compute(s).score;
        var b = M.compute(M.assign(s, (function () { var o = {}; o[f] = s[f] * 1.5; return o; })())).score;
        return b > a;
      });
    } },
    { name: '两国：得分对工时单调递减', run: function () {
      return ['CN', 'ES'].every(function (c) {
        var s = M.defaultState(c);
        return M.compute(M.assign(s, { dailyHours: s.dailyHours + 3 })).score < M.compute(s).score;
      });
    } },
    { name: '两国：远程天数增加能提高得分', run: function () {
      return ['CN', 'ES'].every(function (c) {
        var s = M.assign(M.defaultState(c), { commuteHours: 2, wfhDays: 0 });
        return M.compute(M.assign(s, { wfhDays: 3 })).score > M.compute(s).score;
      });
    } },
    { name: '两国：学历更高 → 同薪得分更低（期望在分母）', run: function () {
      return ['CN', 'ES'].every(function (c) {
        var s = M.assign(M.defaultState(c), { degree: 1, school: 0 });
        return M.compute(M.assign(s, { degree: 3, school: 2 })).score < M.compute(s).score;
      });
    } },
    { name: '两国：风险加点越多 → 得分越低', run: function () {
      return ['CN', 'ES'].every(function (c) {
        var s = M.defaultState(c);
        return M.compute(M.assign(s, { layoff: 3, payDelay: 3, bizHealth: 3 })).score < M.compute(s).score;
      });
    } },
    { name: '两国：税后总包必定不高于税前', run: function () {
      return ['CN', 'ES'].every(function (c) {
        var f = M.salaryFieldOf(c), p = {};
        p[f] = c === 'CN' ? 50000 : 80000;
        var s = M.assign(M.defaultState(c), p);
        return M.compute(M.assign(s, { afterTax: true })).tc <= M.compute(s).tc;
      });
    } },
    { name: '两国：极端输入不产生 NaN / Infinity', run: function () {
      return ['CN', 'ES'].every(function (c) {
        var f = M.salaryFieldOf(c), p = { dailyHours: 24, slackHours: 24, annualLeave: 300,
          publicHolidays: 300, sickLeave: 300, workDaysPerWeek: 0 };
        p[f] = 0;
        var r = M.compute(M.assign(M.defaultState(c), p));
        return isFinite(r.score) && isFinite(r.effHours) && isFinite(r.workDays) && r.effHours > 0;
      });
    } },
    { name: '两国：雷达五维始终落在 0~100', run: function () {
      var ok = true;
      ['CN', 'ES'].forEach(function (c) {
        var f = M.salaryFieldOf(c);
        [{}, (function () { var o = { dailyHours: 5, slackHours: 4 }; o[f] = 1e6; return o; })(),
             (function () { var o = { dailyHours: 16, oncall: 4 }; o[f] = 100; return o; })()
        ].forEach(function (p) {
          var r = M.compute(M.assign(M.defaultState(c), p)).radar;
          Object.keys(r).forEach(function (k) { if (!(r[k] >= 0 && r[k] <= 100)) ok = false; });
        });
      });
      return ok;
    } },
    /* 税后锚修正在「中位收入」上必须精确对齐；收入高于中位时税后得分本来就该更低
     * （累进税让高收入者留存比例更小），所以只在锚点上验精确性。 */
    { name: '税后锚：在中位收入上，税前/税后得分完全一致', run: function () {
      var anchors = { CN: { monthlyBase: 11143, salaryMonths: 14, allowanceMonthly: 0,
                            stockAnnual: 0, fundRate: 2, specialDeduct: 1500, workYears: 0 },
                      ES: { grossAnnual: 22000, bonusCash: 0, allowanceMonthly: 0,
                            stockAnnual: 0, specialDeduct: 0, workYears: 0 } };
      return ['CN', 'ES'].every(function (c) {
        var s = M.assign(M.defaultState(c), anchors[c]);
        var g = M.compute(M.assign(s, { afterTax: false })).score;
        var n = M.compute(M.assign(s, { afterTax: true })).score;
        return Math.abs(n / g - 1) < 0.005;
      });
    } },
    { name: '税后锚：收入高于中位时，税后得分应低于税前（累进税）', run: function () {
      return ['CN', 'ES'].every(function (c) {
        var f = M.salaryFieldOf(c), p = {};
        p[f] = c === 'CN' ? 50000 : 80000;
        var s = M.assign(M.defaultState(c), p);
        return M.compute(M.assign(s, { afterTax: true })).score <
               M.compute(M.assign(s, { afterTax: false })).score;
      });
    } },
    { name: '税后口径不会让西班牙相对中国被额外压分', run: function () {
      // 两国各取中位处境，切税后前后的「得分比」变化应小于 3%
      var cn = M.defaultState('CN'), es = M.defaultState('ES');
      var gRatio = M.compute(M.assign(cn, { afterTax: false })).score /
                   M.compute(M.assign(es, { afterTax: false })).score;
      var nRatio = M.compute(M.assign(cn, { afterTax: true })).score /
                   M.compute(M.assign(es, { afterTax: true })).score;
      return Math.abs(nRatio / gRatio - 1) < 0.03;
    } },
    { name: '未上市期权按面值不会虚增总包', run: function () {
      var r = M.compute(M.assign(M.defaultState('ES'), { stockAnnual: 500000, stockType: 5 }));
      return r.stockValue < 500000 * 0.05 + 1;
    } },
    { name: '诊断结果按提升幅度降序（两国）', run: function () {
      return ['CN', 'ES'].every(function (c) {
        var rows = M.diagnose(M.defaultState(c));
        for (var i = 1; i < rows.length; i++) if (rows[i].delta > rows[i - 1].delta + 1e-9) return false;
        return rows.length > 0;
      });
    } },

    /* ---- 求解器 ---- */
    { name: '求解器：解出的薪资代回去能复现目标得分', run: function () {
      return ['CN', 'ES'].every(function (c) {
        var s = M.defaultState(c);
        var target = 1.35;
        var sol = M.solveSalary(s, target);
        return sol && Math.abs(sol.result.score - target) < 1e-4;
      });
    } },
    { name: '求解器：目标得分越高 → 需要的薪资越高', run: function () {
      return ['CN', 'ES'].every(function (c) {
        var s = M.defaultState(c);
        var a = M.solveSalary(s, 0.8), b = M.solveSalary(s, 1.6);
        return a && b && b.value > a.value;
      });
    } },
    { name: '求解器：税后模式下依然收敛（累进税非线性）', run: function () {
      return ['CN', 'ES'].every(function (c) {
        var s = M.assign(M.defaultState(c), { afterTax: true });
        var sol = M.solveSalary(s, 1.2);
        return sol && Math.abs(sol.result.score - 1.2) < 1e-3;
      });
    } },
    { name: '跨国：追平后两边得分相等', run: function () {
      var cn = M.assign(M.defaultState('CN'), { afterTax: true });
      var es = M.assign(M.defaultState('ES'), { afterTax: true });
      var x = M.crossCompare(cn, es);
      return x.bNeedsToMatchA && x.aNeedsToMatchB &&
             Math.abs(x.bNeedsToMatchA.result.score - x.a.score) < 1e-3 &&
             Math.abs(x.aNeedsToMatchB.result.score - x.b.score) < 1e-3;
    } },
    { name: '差异归因：七个因子连乘 = 得分比', run: function () {
      var cn = M.assign(M.defaultState('CN'), { afterTax: true, city: 0, workYears: 3 });
      var es = M.assign(M.defaultState('ES'), { afterTax: true, city: 0, workYears: 3 });
      var at = M.attribution(M.compute(cn), M.compute(es));
      return Math.abs(at.product - at.ratio) < 1e-9;
    } },
    { name: '换国家保留通用项、重置国家专属项', run: function () {
      var s = M.assign(M.defaultState('CN'), { techStack: 0, leader: 0, monthlyBase: 99999, city: 0 });
      var t = M.switchCountry(s, 'ES');
      return t.country === 'ES' && t.techStack === 0 && t.leader === 0 &&
             t.grossAnnual === D.COUNTRIES.ES.defaults.grossAnnual &&
             t.city === D.COUNTRIES.ES.defaults.city;
    } },
    { name: 'PPP 标准化：西班牙 1 欧 ≈ 6.76 元购买力', run: function () {
      var r = M.compute(M.defaultState('ES'));
      return Math.abs(r.pppMul - 4.19 / 0.62) < 1e-6;
    } }
  ];

  /* ------------------------------------------------------------------ */
  function run() {
    var res = { cases: [], nets: [], invariants: [], pass: 0, fail: 0 };

    CASES.forEach(function (c) {
      var r = M.compute(c.state);
      var ok = r.score >= c.expect[0] && r.score <= c.expect[1];
      res.cases.push({
        cc: c.cc, flag: D.COUNTRIES[c.cc].flag, name: c.name, note: c.note,
        score: r.score, rating: r.rating.label, expect: c.expect, ok: ok,
        detail: { cur: r.cur, tc: r.tc, workDays: r.workDays, dailySalary: r.dailySalary,
                  effHours: r.effHours, rawHourly: r.rawHourly, pppAdj: r.pppAdjHourly,
                  env: r.envFactor, growth: r.growthFactor, expect: r.expectFactor, risk: r.riskFactor }
      });
      ok ? res.pass++ : res.fail++;
    });

    NET_CHECKS.forEach(function (n) {
      var f = M.salaryFieldOf(n.cc), p = { afterTax: true };
      p[f] = n.gross;
      var r = M.compute(M.assign(M.defaultState(n.cc), p));
      var ok = r.tcNet >= n.expectNet[0] && r.tcNet <= n.expectNet[1];
      res.nets.push({ label: n.label, net: r.tcNet, monthly: r.tcNet / 12,
                      ss: r.personalSocial, tax: r.tax, rate: r.effectiveTaxRate,
                      expect: n.expectNet, ok: ok });
      ok ? res.pass++ : res.fail++;
    });

    INVARIANTS.forEach(function (inv) {
      var ok = false, err = null;
      try { ok = !!inv.run(); } catch (e) { err = e.message; }
      res.invariants.push({ name: inv.name, ok: ok, err: err });
      ok ? res.pass++ : res.fail++;
    });

    return res;
  }

  root.PJC_TESTS = { run: run, CASES: CASES, INVARIANTS: INVARIANTS, NET_CHECKS: NET_CHECKS };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = root.PJC_TESTS;
    if (require.main === module) {
      var r = run(), f = M.fmt;
      console.log('\n=== 标定用例 ===');
      r.cases.forEach(function (c) {
        console.log((c.ok ? '  PASS ' : '  FAIL ') + c.score.toFixed(3).padStart(6) +
          '  [' + c.expect[0] + ', ' + c.expect[1] + ']  ' + c.rating.padEnd(6) + '  ' + c.flag + ' ' + c.name);
        console.log('        TC=' + c.detail.cur + Math.round(c.detail.tc) +
          ' 工作日=' + c.detail.workDays.toFixed(1) +
          ' 有效工时=' + c.detail.effHours.toFixed(2) +
          ' 名义时薪=' + c.detail.cur + Math.round(c.detail.rawHourly) +
          ' PPP实感时薪=¥' + Math.round(c.detail.pppAdj) +
          ' | env=' + c.detail.env.toFixed(3) + ' growth=' + c.detail.growth.toFixed(3) +
          ' expect=' + c.detail.expect.toFixed(3) + ' risk=' + c.detail.risk.toFixed(3));
      });
      console.log('\n=== 税后口径校验 ===');
      r.nets.forEach(function (n) {
        console.log((n.ok ? '  PASS  ' : '  FAIL  ') + n.label + ' ' + Math.round(n.net) +
          '（月 ' + Math.round(n.monthly) + '）  SS=' + Math.round(n.ss) + ' 税=' + Math.round(n.tax) +
          ' 有效税负=' + (n.rate * 100).toFixed(1) + '%   期望 ' + n.expect[0] + '~' + n.expect[1]);
      });
      console.log('\n=== 不变量 ===');
      r.invariants.forEach(function (i) {
        console.log((i.ok ? '  PASS  ' : '  FAIL  ') + i.name + (i.err ? '  (' + i.err + ')' : ''));
      });
      console.log('\n通过 ' + r.pass + ' / 失败 ' + r.fail + '\n');
      process.exit(r.fail ? 1 : 0);
    }
  }

})(typeof window !== 'undefined' ? window : globalThis);
