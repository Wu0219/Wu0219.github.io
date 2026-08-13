/* ===========================================================================
 * app.js —— UI 渲染与交互（国际版）
 * 表单由 data.js 的 buildSections(country) 声明式渲染；计算全部委托给 model.js
 * =========================================================================== */

(function () {
  'use strict';

  var D = window.PJC_DATA, M = window.PJC_MODEL, F = M.fmt;
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var esc = function (s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  };

  var LS_STATE = 'pjc.state.v2';
  var LS_OFFERS = 'pjc.offers.v2';
  var LS_DUAL = 'pjc.dual.v2';
  var LS_THEME = 'pjc.theme';
  var LS_OPEN = 'pjc.open.v2';

  var state = M.defaultState('CN');
  var offers = [];
  var dual = { CN: null, ES: null };   // 跨国对照的两个槽位

  function C() { return D.COUNTRIES[state.country]; }
  function cc(code) { return D.COUNTRIES[code]; }
  function cur(code) { return D.COUNTRIES[code].cur; }

  /* =====================================================================
   * 预设场景（按国家）
   * ===================================================================== */
  function coIdx(code, key) {
    var l = D.COUNTRIES[code].companyTypes;
    for (var i = 0; i < l.length; i++) if (l[i].key === key) return i;
    return 0;
  }
  function optIdx(dims, dimKey, needle) {
    for (var i = 0; i < dims.length; i++) if (dims[i].key === dimKey) {
      for (var j = 0; j < dims[i].options.length; j++)
        if (dims[i].options[j].label.indexOf(needle) >= 0) return j;
    }
    return 0;
  }

  var PRESETS = {
    CN: [
      { name: '大厂应届', patch: { monthlyBase: 20000, salaryMonths: 15, fundRate: 4, city: 0,
        companyType: coIdx('CN', 'bigtech'), dailyHours: 10.5, commuteHours: 1.5, commuteComfort: 3,
        slackHours: 1.5, oncall: 1, annualLeave: 10, leaveHard: 1, workYears: 0, age: 23, degree: 1, school: 2 } },
      { name: '大厂 3~5 年', patch: { monthlyBase: 35000, salaryMonths: 16, stockAnnual: 80000, stockType: 1,
        fundRate: 4, city: 0, companyType: coIdx('CN', 'bigtech'), dailyHours: 10.5, commuteHours: 1.5,
        commuteComfort: 3, slackHours: 1.5, oncall: 1, annualLeave: 10, leaveHard: 1, workYears: 2, age: 28 } },
      { name: '国企 / 银行', patch: { monthlyBase: 15000, salaryMonths: 14, fundRate: 4, city: 3,
        companyType: coIdx('CN', 'soe'), dailyHours: 8.5, commuteHours: 1, slackHours: 2, oncall: 0,
        annualLeave: 10, leaveHard: 0, workYears: 3, age: 31,
        techStack: optIdx(D.GROWTH_DIMS, 'techStack', '传统企业开发'),
        autonomy: optIdx(D.GROWTH_DIMS, 'autonomy', '改配置'),
        codebase: optIdx(D.GROWTH_DIMS, 'codebase', '屎山') } },
      { name: 'AI 创业', patch: { monthlyBase: 45000, salaryMonths: 14, stockAnnual: 300000, stockType: 4,
        fundRate: 2, city: 0, companyType: coIdx('CN', 'startup'), dailyHours: 11, commuteHours: 1,
        slackHours: 1, oncall: 1, annualLeave: 10, leaveHard: 1, workYears: 2, age: 29,
        degree: 2, school: 2, bachelorSchool: 2,
        techStack: optIdx(D.GROWTH_DIMS, 'techStack', 'AI / 大模型'),
        autonomy: optIdx(D.GROWTH_DIMS, 'autonomy', '核心自研'),
        codebase: optIdx(D.GROWTH_DIMS, 'codebase', '新项目'),
        bizProspect: optIdx(D.GROWTH_DIMS, 'bizProspect', '高增长') } },
      { name: '外包驻场 996', patch: { monthlyBase: 18000, salaryMonths: 12, fundRate: 0, city: 3,
        companyType: coIdx('CN', 'outsource'), workDaysPerWeek: 4, dailyHours: 12, commuteHours: 1.5,
        commuteComfort: 3, slackHours: 0.5, annualLeave: 5, leaveHard: 2, workYears: 2, age: 29, contract: 2,
        techStack: optIdx(D.GROWTH_DIMS, 'techStack', '传统企业开发'),
        autonomy: optIdx(D.GROWTH_DIMS, 'autonomy', '外派驻场'),
        codebase: optIdx(D.GROWTH_DIMS, 'codebase', '考古现场'),
        engineering: optIdx(D.GROWTH_DIMS, 'engineering', '全靠手动'),
        leader: optIdx(D.ENV_DIMS, 'leader', '外行指挥'),
        office: optIdx(D.ENV_DIMS, 'office', '无固定工位') } }
    ],
    ES: [
      { name: 'Junior producto', patch: { grossAnnual: 26000, city: 0, companyType: coIdx('ES', 'product'),
        dailyHours: 8.5, commuteHours: 1, wfhDays: 2, slackHours: 1.5, annualLeave: 23, leaveHard: 0,
        workYears: 0, age: 24 } },
      { name: 'Senior producto', patch: { grossAnnual: 55000, city: 0, companyType: coIdx('ES', 'product'),
        dailyHours: 8.5, commuteHours: 1, wfhDays: 2, slackHours: 1.5, annualLeave: 23, leaveHard: 0,
        workYears: 3, age: 33,
        engineering: optIdx(D.GROWTH_DIMS, 'engineering', 'CR + 单测'),
        autonomy: optIdx(D.GROWTH_DIMS, 'autonomy', '参与核心') } },
      { name: 'Big Tech Madrid', patch: { grossAnnual: 90000, stockAnnual: 25000, stockType: 1, city: 0,
        companyType: coIdx('ES', 'bigtech'), dailyHours: 8.5, commuteHours: 1, wfhDays: 2, slackHours: 1.5,
        annualLeave: 25, leaveHard: 0, workYears: 3, age: 33, oncall: 1,
        techStack: optIdx(D.GROWTH_DIMS, 'techStack', '基础架构'),
        engineering: optIdx(D.GROWTH_DIMS, 'engineering', 'CR + 单测'),
        team: optIdx(D.ENV_DIMS, 'team', '大牛云集') } },
      { name: 'Cárnica / 外派', patch: { grossAnnual: 38000, city: 0, companyType: coIdx('ES', 'carnica'),
        dailyHours: 9, commuteHours: 1.25, wfhDays: 0, slackHours: 1, annualLeave: 22, leaveHard: 1,
        workYears: 3, age: 33, contract: 2,
        techStack: optIdx(D.GROWTH_DIMS, 'techStack', '传统企业开发'),
        autonomy: optIdx(D.GROWTH_DIMS, 'autonomy', '外派驻场'),
        codebase: optIdx(D.GROWTH_DIMS, 'codebase', '屎山'),
        engineering: optIdx(D.GROWTH_DIMS, 'engineering', '全靠手动'),
        office: optIdx(D.ENV_DIMS, 'office', '无固定工位') } },
      { name: '银行 IT 955', patch: { grossAnnual: 45000, city: 0, companyType: coIdx('ES', 'banca'),
        dailyHours: 8, commuteHours: 1, wfhDays: 2, slackHours: 2, annualLeave: 25, leaveHard: 0,
        workYears: 4, age: 38,
        techStack: optIdx(D.GROWTH_DIMS, 'techStack', '传统企业开发'),
        codebase: optIdx(D.GROWTH_DIMS, 'codebase', '屎山') } },
      { name: '100% 远程', patch: { grossAnnual: 50000, city: 7, companyType: coIdx('ES', 'product'),
        dailyHours: 8, commuteHours: 0, wfhDays: 5, slackHours: 1.5, annualLeave: 24, leaveHard: 0,
        workYears: 3, age: 32 } }
    ]
  };

  /* =====================================================================
   * 持久化
   * ===================================================================== */
  function save() {
    try {
      localStorage.setItem(LS_STATE, JSON.stringify(state));
      localStorage.setItem(LS_OFFERS, JSON.stringify(offers));
      localStorage.setItem(LS_DUAL, JSON.stringify(dual));
    } catch (e) {}
  }
  function load() {
    try {
      var s = localStorage.getItem(LS_STATE);
      if (s) {
        var o = JSON.parse(s);
        state = M.assign(M.defaultState(o.country || 'CN'), o);
      }
      var of = localStorage.getItem(LS_OFFERS);
      if (of) offers = JSON.parse(of) || [];
      var du = localStorage.getItem(LS_DUAL);
      if (du) dual = M.assign({ CN: null, ES: null }, JSON.parse(du) || {});
    } catch (e) {}
  }
  function encodeState(s) {
    try { return btoa(unescape(encodeURIComponent(JSON.stringify(s)))); } catch (e) { return ''; }
  }
  function decodeState(x) {
    try { return JSON.parse(decodeURIComponent(escape(atob(x)))); } catch (e) { return null; }
  }
  function readHash() {
    var m = (location.hash || '').match(/[#&]s=([^&]+)/);
    return m ? decodeState(decodeURIComponent(m[1])) : null;
  }

  /* =====================================================================
   * 表单
   * ===================================================================== */
  function sections() { return D.buildSections(state.country); }

  function depVisible(f) {
    if (!f.dep) return true;
    if (f.dep === 'afterTax') return !!state.afterTax;
    if (f.dep === 'isMaster') return (state.degree | 0) === 2;
    return true;
  }

  function fieldHTML(f) {
    var id = 'f_' + f.key, body = '';
    if (f.type === 'number') {
      body = '<div class="with-unit"><input type="number" id="' + id + '" data-key="' + f.key + '"' +
        (f.min !== undefined ? ' min="' + f.min + '"' : '') +
        (f.max !== undefined ? ' max="' + f.max + '"' : '') +
        (f.step !== undefined ? ' step="' + f.step + '"' : '') +
        ' value="' + state[f.key] + '">' +
        (f.unit ? '<span class="unit">' + esc(f.unit) + '</span>' : '') + '</div>';
    } else if (f.type === 'select') {
      body = '<select id="' + id + '" data-key="' + f.key + '">' + f.options.map(function (o) {
        return '<option value="' + o.value + '"' + (state[f.key] == o.value ? ' selected' : '') + '>' +
               esc(o.label) + '</option>';
      }).join('') + '</select><div class="opt-hint" data-opthint="' + f.key + '"></div>';
    } else if (f.type === 'toggle') {
      body = '<label class="switch"><input type="checkbox" data-key="' + f.key + '"' +
        (state[f.key] ? ' checked' : '') + '><span class="track"></span>' +
        '<span style="font-size:12px;color:var(--text-dim)">' + esc(f.label) + '</span></label>';
    } else if (f.type === 'checks') {
      var sel = Array.isArray(state[f.key]) ? state[f.key] : [];
      body = '<div class="checks">' + f.options.map(function (o) {
        return '<label class="chk"><input type="checkbox" data-checkkey="' + f.key + '" value="' + o.value + '"' +
          (sel.indexOf(o.value) >= 0 ? ' checked' : '') + '>' + esc(o.label) + '</label>';
      }).join('') + '</div>';
    }
    return '<div class="field' + (f.type === 'checks' || f.type === 'toggle' ? ' full' : '') +
      '" data-field="' + f.key + '">' +
      (f.type !== 'toggle' ? '<label class="lb" for="' + id + '">' + esc(f.label) + '</label>' : '') +
      body + (f.hint ? '<div class="hint">' + esc(f.hint) + '</div>' : '') + '</div>';
  }

  function renderForm() {
    var openMap = {};
    try { openMap = JSON.parse(localStorage.getItem(LS_OPEN) || '{}'); } catch (e) {}
    $('#form').innerHTML = sections().map(function (sec, i) {
      var open = openMap[sec.id] !== undefined ? openMap[sec.id] : (i < 3);
      return '<div class="card section' + (open ? ' open' : '') + '" data-sec="' + sec.id + '">' +
        '<div class="section-head"><div class="ico">' + esc(sec.icon) + '</div>' +
        '<div><h2>' + esc(sec.title) + '</h2><div class="desc">' + esc(sec.desc) + '</div></div>' +
        '<div class="chev">▶</div></div>' +
        '<div class="section-body"><div class="grid2">' + sec.fields.map(fieldHTML).join('') +
        '</div></div></div>';
    }).join('');
  }

  function syncFormFromState() {
    sections().forEach(function (sec) {
      sec.fields.forEach(function (f) {
        if (f.type === 'checks') {
          $$('[data-checkkey="' + f.key + '"]').forEach(function (c) {
            c.checked = (state[f.key] || []).indexOf(c.value) >= 0;
          });
          return;
        }
        var el = $('[data-key="' + f.key + '"]');
        if (!el) return;
        if (f.type === 'toggle') el.checked = !!state[f.key]; else el.value = state[f.key];
      });
    });
  }

  function updateDeps() {
    sections().forEach(function (sec) {
      sec.fields.forEach(function (f) {
        var box = $('[data-field="' + f.key + '"]');
        if (box) box.style.display = depVisible(f) ? '' : 'none';
        if (f.type === 'select') {
          var el = $('[data-opthint="' + f.key + '"]');
          if (!el) return;
          var o = f.options.filter(function (x) { return x.value == state[f.key]; })[0];
          el.textContent = (o && o.hint) ? o.hint : '';
          el.style.display = (o && o.hint) ? '' : 'none';
        }
      });
    });
  }

  /* =====================================================================
   * 结果面板
   * ===================================================================== */
  function renderScore(r) {
    var rt = r.rating;
    $('#scoreNum').textContent = isFinite(r.score) ? r.score.toFixed(2) : '—';
    $('#scoreNum').style.color = rt.color;
    var b = $('#scoreRating');
    b.innerHTML = '<span>' + rt.emoji + '</span><span>' + esc(rt.label) + '</span>';
    b.style.background = rt.color;
    $('#scoreDesc').textContent = rt.desc;
    $('#scoreScale').innerHTML = D.RATINGS.map(function (x) {
      return '<div class="seg' + (x.key === rt.key ? ' on' : '') + '" style="background:' + x.color +
             '" title="' + esc(x.label) + '"></div>';
    }).join('');
    $('#adjHourly').textContent = r.cur + F.money(r.adjHourly);
    $('#deservedHourly').textContent = r.cur + F.money(r.deservedHourly);
    $('#adjHourly').style.color = r.adjHourly >= r.deservedHourly ? 'var(--accent)' : 'var(--danger)';
    $('#scoreCountry').textContent = r.country.flag + ' ' + r.country.name + ' 本地刻度';
  }

  function radarSVG(radar) {
    var dims = D.RADAR_DIMS, n = dims.length, cx = 150, cy = 142, R = 94;
    function pt(i, rr) { var a = -Math.PI / 2 + i * 2 * Math.PI / n; return [cx + Math.cos(a) * rr, cy + Math.sin(a) * rr]; }
    function poly(rs) { return dims.map(function (_, i) { return pt(i, rs[i]).map(function (v) { return v.toFixed(1); }).join(','); }).join(' '); }
    var out = '<svg viewBox="0 0 300 288" width="100%" role="img" aria-label="五维雷达图">';
    [0.25, 0.5, 0.75, 1].forEach(function (k) {
      out += '<polygon points="' + poly(dims.map(function () { return R * k; })) +
        '" fill="none" stroke="currentColor" stroke-width="1" opacity="' + (k === 1 ? .35 : .16) + '"/>';
    });
    dims.forEach(function (d, i) {
      var p = pt(i, R);
      out += '<line x1="' + cx + '" y1="' + cy + '" x2="' + p[0].toFixed(1) + '" y2="' + p[1].toFixed(1) +
        '" stroke="currentColor" stroke-width="1" opacity=".18"/>';
    });
    var vals = dims.map(function (d) { return Math.max(0, Math.min(100, radar[d.key])); });
    out += '<polygon points="' + poly(vals.map(function (v) { return R * v / 100; })) +
      '" fill="var(--accent)" fill-opacity=".20" stroke="var(--accent)" stroke-width="2" stroke-linejoin="round"/>';
    dims.forEach(function (d, i) {
      var p = pt(i, R * vals[i] / 100);
      out += '<circle cx="' + p[0].toFixed(1) + '" cy="' + p[1].toFixed(1) + '" r="3.2" fill="var(--accent)"/>';
    });
    dims.forEach(function (d, i) {
      var p = pt(i, R + 22);
      var an = Math.abs(p[0] - cx) < 3 ? 'middle' : (p[0] > cx ? 'start' : 'end');
      if (an === 'start') p[0] -= 6; if (an === 'end') p[0] += 6;
      out += '<text x="' + p[0].toFixed(1) + '" y="' + (p[1] + 4).toFixed(1) + '" text-anchor="' + an +
        '" font-size="12" fill="currentColor" opacity=".72">' + esc(d.label) + '</text>';
    });
    return out + '</svg>';
  }

  function renderRadar(r) {
    $('#radar').innerHTML = radarSVG(r.radar);
    $('#radarLegend').innerHTML = D.RADAR_DIMS.map(function (d) {
      return '<div title="' + esc(d.hint) + '"><div class="rl-v">' + Math.round(r.radar[d.key]) +
        '</div><div class="rl-k">' + esc(d.label) + '</div></div>';
    }).join('');
  }

  function renderKV(r) {
    var code = r.country.key;
    var items = [
      ['年总包（计入公式）', r.cur + F.wan(r.tc, code), state.afterTax ? '税后口径' : '税前口径'],
      ['年实际工作天数', r.workDays.toFixed(1), '天'],
      ['有效日薪', r.cur + F.money(r.dailySalary), '每工作日'],
      ['有效工时', r.effHours.toFixed(2), '小时/天'],
      ['名义时薪', r.cur + F.money(r.rawHourly), '日薪 ÷ 有效工时'],
      ['PPP 标准化时薪', '¥' + F.money(r.pppRawHourly), '折算成中国购买力']
    ];
    $('#kv').innerHTML = items.map(function (it) {
      return '<div class="kv"><div class="k">' + esc(it[0]) + '</div><div class="v">' +
        esc(it[1]) + '<small>' + esc(it[2]) + '</small></div></div>';
    }).join('');
  }

  function row(k, v, cls) {
    return '<div class="frow"><span class="fk">' + k + '</span><span class="fv ' + (cls || '') + '">' + v + '</span></div>';
  }

  function renderFormula(r) {
    var b = r.breakdown, h = '';
    h += '<div class="ftitle">分子 · 你收获了什么</div>';
    h += row('标准化日薪', r.cur + F.money(r.dailySalary));
    h += row('× 环境系数', r.envFactor.toFixed(3), 'up');
    h += row('&nbsp;&nbsp;├ 城市生活成本', r.cityFactor.toFixed(2));
    h += row('&nbsp;&nbsp;├ 人文环境加权', r.envHuman.toFixed(3));
    h += row('&nbsp;&nbsp;└ 福利加分', '+' + r.perkBonus.toFixed(3));
    h += row('× 成长系数', r.growthFactor.toFixed(3), 'up');
    h += '<hr>' + row('分子合计', F.money(r.numerator), 'up total');
    h += '<div class="ftitle" style="margin-top:14px">分母 · 你付出了什么 + 本该拿多少</div>';
    h += row('基准时薪（' + r.country.name + '）', r.cur + r.baseHourly);
    h += row('× 有效工时', r.effHours.toFixed(2) + ' h', 'down');
    h += row('&nbsp;&nbsp;├ 标准工时', b.baseH.toFixed(2));
    h += row('&nbsp;&nbsp;├ 加班（折算后）', '+' + b.otH.toFixed(2));
    h += row('&nbsp;&nbsp;├ 有效通勤', '+' + b.commuteH.toFixed(2));
    h += row('&nbsp;&nbsp;├ On-call 折算', '+' + b.oncallH.toFixed(2));
    h += row('&nbsp;&nbsp;└ 摸鱼折回', '−' + b.slackH.toFixed(2));
    h += row('× 期望系数', r.expectFactor.toFixed(3), 'down');
    h += row('&nbsp;&nbsp;├ 学历系数', r.eduFactor.toFixed(2));
    h += row('&nbsp;&nbsp;└ 年限×赛道', r.yearFactor.toFixed(2));
    h += row('× 风险系数', r.riskFactor.toFixed(3), 'down');
    h += row('&nbsp;&nbsp;├ 公司基础风险', r.company.risk.toFixed(2));
    h += row('&nbsp;&nbsp;├ 风险加点合计', '+' + (r.riskAdds.reduce(function (a, x) { return a + x; }, 0) + r.contractAdd).toFixed(2));
    h += row('&nbsp;&nbsp;└ 年龄风险', '+' + r.ageAdd.toFixed(3));
    h += '<hr>' + row('分母合计', F.money(r.denominator), 'down total');
    h += '<hr>' + row('<b>工作性价比</b>', '<b>' + r.score.toFixed(3) + '</b>', 'total');
    if (state.afterTax) {
      h += '<div class="ftitle" style="margin-top:14px">税与社保</div>';
      h += row('税前现金 + 股票', r.cur + F.money(r.cashAnnual + r.stockValue));
      h += row('− 个人社保', r.cur + F.money(r.personalSocial));
      h += row('− ' + r.country.labels.taxName, r.cur + F.money(r.tax));
      if (r.companyFund) h += row('+ 公司公积金', r.cur + F.money(r.companyFund));
      h += row('有效税负率', (r.effectiveTaxRate * 100).toFixed(1) + '%', 'down');
    }
    $('#formula').innerHTML = h;
  }

  function renderDiag() {
    var rows = M.diagnose(state, 6);
    if (!rows.length) { $('#diag').innerHTML = '<div class="offer-empty">已经没有可以再优化的项了。</div>'; return; }
    var max = rows[0].delta;
    $('#diag').innerHTML = rows.map(function (d) {
      return '<div class="diag-row">' +
        '<div class="dl"><span class="badge">' + esc(d.group) + '</span>' + esc(d.label) + '</div>' +
        '<div class="dv">' + F.pct(d.pct) + '</div>' +
        (d.target ? '<div class="dt">→ ' + esc(d.target) + '（得分 ' + d.to.toFixed(2) + '）</div>'
                  : '<div class="dt">改善后得分 ' + d.to.toFixed(2) + '</div>') +
        '<div class="diag-bar"><i style="width:' + (d.delta / max * 100).toFixed(1) + '%"></i></div></div>';
    }).join('');
  }

  function renderSensitivity() {
    var rows = M.sensitivity(state).filter(function (r) { return r.span > 1e-6; }).slice(0, 18);
    if (!rows.length) { $('#sens').innerHTML = '<div class="offer-empty">暂无数据</div>'; return; }
    var maxAbs = 0;
    rows.forEach(function (r) { maxAbs = Math.max(maxAbs, Math.abs(r.up), Math.abs(r.down)); });
    $('#sens').innerHTML =
      '<div style="font-size:12px;color:var(--text-dim);margin-bottom:12px">以当前得分 <b class="mono">' +
      rows[0].base.toFixed(2) + '</b> 为基线，把每一项分别拨到最好 / 最差，看得分怎么动。条越长 = 这一项对你的处境影响越大。</div>' +
      rows.map(function (r) {
        var pw = Math.abs(r.up) / maxAbs * 50, nw = Math.abs(r.down) / maxAbs * 50;
        return '<div class="tornado-row"><div class="tl"><span class="badge">' + esc(r.group) + '</span>' +
          esc(r.label) + '</div><div class="tornado-track"><div class="axis"></div>' +
          '<i class="neg" style="width:' + nw.toFixed(1) + '%"></i>' +
          '<i class="pos" style="width:' + pw.toFixed(1) + '%"></i></div>' +
          '<div class="tv">' + r.lo.toFixed(2) + ' ~ ' + r.hi.toFixed(2) + '</div></div>';
      }).join('');
  }

  /* =====================================================================
   * Offer 对比
   * ===================================================================== */
  var OFFER_ROWS = [
    { k: '国家 / 地区', get: function (r) { return r.country.flag + ' ' + r.country.name; } },
    { k: '得分（本地刻度）', get: function (r) { return r.score.toFixed(2); }, cmp: function (r) { return r.score; } },
    { k: '评级', get: function (r) { return r.rating.emoji + ' ' + r.rating.label; } },
    { k: '年总包（本币）', get: function (r) { return r.cur + F.wan(r.tc, r.country.key); } },
    { k: 'PPP 标准化年包', get: function (r) { return '¥' + F.wan(r.pppTC, 'CN'); }, cmp: function (r) { return r.pppTC; } },
    { k: '年工作天数', get: function (r) { return r.workDays.toFixed(0) + ' 天'; }, cmp: function (r) { return -r.workDays; } },
    { k: '有效工时/天', get: function (r) { return r.effHours.toFixed(2) + ' h'; }, cmp: function (r) { return -r.effHours; } },
    { k: '年被占用工时', get: function (r) { return F.money(r.annualHours) + ' h'; }, cmp: function (r) { return -r.annualHours; } },
    { k: 'PPP 名义时薪', get: function (r) { return '¥' + F.money(r.pppRawHourly); }, cmp: function (r) { return r.pppRawHourly; } },
    { k: 'PPP 实感时薪', get: function (r) { return '¥' + F.money(r.pppAdjHourly); }, cmp: function (r) { return r.pppAdjHourly; } },
    { k: '成长系数', get: function (r) { return r.growthFactor.toFixed(3); }, cmp: function (r) { return r.growthFactor; } },
    { k: '环境系数', get: function (r) { return r.envFactor.toFixed(3); }, cmp: function (r) { return r.envFactor; } },
    { k: '期望系数', get: function (r) { return r.expectFactor.toFixed(3); } },
    { k: '风险系数', get: function (r) { return r.riskFactor.toFixed(3); }, cmp: function (r) { return -r.riskFactor; } }
  ];

  function renderOffers() {
    var box = $('#offers');
    if (!offers.length) {
      box.innerHTML = '<div class="offer-empty">还没有保存任何 Offer。<br>在左侧填好一份，点顶部的「存为 Offer」，' +
        '就能横向对比了（最多 6 份，可以混着存中国和西班牙的）。</div>';
      return;
    }
    var res = offers.map(function (o) { return M.compute(o.state); });
    var html = '<div class="note info" style="margin-bottom:12px">跨国比较请看 <b>PPP 标准化</b> 那几行 —— ' +
      '「得分」是各自国家的本地刻度，不能直接横向比大小；PPP 行才是同一把尺子。</div>' +
      '<div class="tbl-scroll"><table class="tbl"><thead><tr><th>指标</th>' +
      offers.map(function (o, i) {
        return '<th class="num">' + esc(o.name) +
          ' <button class="ghost" data-del="' + i + '" style="padding:0 5px;font-size:11px;line-height:1.4">✕</button></th>';
      }).join('') + '</tr></thead><tbody>';
    OFFER_ROWS.forEach(function (rw) {
      var best = -1;
      if (rw.cmp && res.length > 1) {
        var bv = -Infinity;
        res.forEach(function (r, i) { var v = rw.cmp(r); if (v > bv) { bv = v; best = i; } });
      }
      html += '<tr><td>' + esc(rw.k) + '</td>' + res.map(function (r, i) {
        return '<td class="num' + (i === best ? ' best' : '') + '">' + rw.get(r) + '</td>';
      }).join('') + '</tr>';
    });
    html += '</tbody></table></div><div style="margin-top:12px">' +
      '<button class="ghost" id="clearOffers">清空全部</button> ' +
      '<button class="ghost" id="loadFirst">把第 1 份载入编辑区</button></div>';
    box.innerHTML = html;
    $$('[data-del]', box).forEach(function (b) {
      b.onclick = function () { offers.splice(+b.dataset.del, 1); save(); renderOffers(); };
    });
    var c = $('#clearOffers'); if (c) c.onclick = function () { offers = []; save(); renderOffers(); };
    var l = $('#loadFirst'); if (l) l.onclick = function () { loadState(offers[0].state); };
  }

  /* =====================================================================
   * 🌍 跨国对照 + 等效薪资求解
   * ===================================================================== */
  function slotSummary(code) {
    var s = dual[code];
    var C0 = cc(code);
    if (!s) {
      return '<div class="dual-slot empty"><div class="dual-flag">' + C0.flag + '</div>' +
        '<div class="dual-name">' + esc(C0.name) + '</div>' +
        '<div class="offer-empty" style="padding:14px 6px">还没设定。<br>把左边编辑区切到' + esc(C0.name) +
        '、填好之后点下面的按钮存进来。</div>' +
        '<button class="primary" data-dualset="' + code + '">用当前编辑区填入</button></div>';
    }
    var r = M.compute(s);
    return '<div class="dual-slot"><div class="dual-flag">' + C0.flag + '</div>' +
      '<div class="dual-name">' + esc(C0.name) + '</div>' +
      '<div class="dual-score" style="color:' + r.rating.color + '">' + r.score.toFixed(2) + '</div>' +
      '<div class="dual-rating">' + r.rating.emoji + ' ' + esc(r.rating.label) + '</div>' +
      '<div class="dual-kv">' +
        kvline('年总包', r.cur + F.wan(r.tc, code) + (s.afterTax ? '（税后）' : '（税前）')) +
        kvline('PPP 标准化年包', '¥' + F.wan(r.pppTC, 'CN')) +
        kvline('年工作天数', r.workDays.toFixed(0) + ' 天') +
        kvline('有效工时', r.effHours.toFixed(2) + ' h/天') +
        kvline('PPP 实感时薪', '¥' + F.money(r.pppAdjHourly)) +
        kvline('公司类型', r.company.label.split('（')[0]) +
        kvline('工作年限', C0.years[s.workYears | 0].label) +
      '</div>' +
      '<div class="dual-btns">' +
        '<button data-dualset="' + code + '">用当前编辑区覆盖</button>' +
        '<button data-dualload="' + code + '">载入编辑区</button>' +
        '<button class="ghost" data-dualclear="' + code + '">清空</button>' +
      '</div></div>';
  }
  function kvline(k, v) {
    return '<div class="dual-line"><span>' + esc(k) + '</span><b>' + esc(v) + '</b></div>';
  }

  function renderDual() {
    var box = $('#dual');
    var head = '<div class="note info" style="margin-bottom:14px">' +
      '把中国和西班牙各存一份，就能算出 <b>「另一边要拿多少钱，才能追平这一边」</b>。' +
      '两边的税制、社保、假期、薪资曲线、年龄风险都按各自国家计算，钱通过 PPP 购买力换算对齐。' +
      '<br>建议两边都打开「按税后口径计算」—— 西班牙的税负明显重于中国，税前比较会高估西班牙的 offer。</div>';

    var slots = '<div class="dual-grid">' + slotSummary('CN') + slotSummary('ES') + '</div>';

    var body = '';
    if (dual.CN && dual.ES) {
      var x = M.crossCompare(dual.CN, dual.ES);
      var rA = x.a, rB = x.b;

      /* ---- 等效薪资求解 ---- */
      var solveB = x.bNeedsToMatchA, solveA = x.aNeedsToMatchB;
      body += '<h3 class="dual-h3">等效薪资：另一边要拿多少才追得平</h3>';
      body += '<div class="solve-grid">';

      if (solveB) {
        var esNow = rB.tc, esNeed = solveB.annual;
        var gapB = esNeed - (cc('ES').salaryMode === 'monthly' ? 0 : (dual.ES.grossAnnual || 0));
        body += '<div class="solve-card">' +
          '<div class="solve-head">🇪🇸 西班牙要拿到</div>' +
          '<div class="solve-num">€' + F.money(esNeed) + '</div>' +
          '<div class="solve-sub">bruto anual（税前年薪），才能追平中国这份工作的 <b>' + rA.score.toFixed(2) + '</b> 分</div>' +
          '<div class="solve-delta ' + (gapB > 0 ? 'up' : 'down') + '">' +
            '当前 €' + F.money(dual.ES.grossAnnual || 0) + ' → 需要 ' +
            (gapB > 0 ? '再涨 €' + F.money(gapB) + '（+' + (dual.ES.grossAnnual ? (gapB / dual.ES.grossAnnual * 100).toFixed(0) : '∞') + '%）'
                      : '已经超出 €' + F.money(-gapB)) +
          '</div></div>';
      }
      if (solveA) {
        var cnNeedM = solveA.value, cnNeedY = solveA.annual;
        var cnNowM = dual.CN.monthlyBase || 0;
        var gapA = cnNeedM - cnNowM;
        body += '<div class="solve-card">' +
          '<div class="solve-head">🇨🇳 中国要拿到</div>' +
          '<div class="solve-num">¥' + F.money(cnNeedM) + '<small> / 月</small></div>' +
          '<div class="solve-sub">即年总包约 <b>¥' + F.wan(cnNeedY, 'CN') + '</b>（按当前 ' +
            (dual.CN.salaryMonths || 12) + ' 薪结构等比缩放），才能追平西班牙这份工作的 <b>' + rB.score.toFixed(2) + '</b> 分</div>' +
          '<div class="solve-delta ' + (gapA > 0 ? 'up' : 'down') + '">' +
            '当前 ¥' + F.money(cnNowM) + '/月 → 需要 ' +
            (gapA > 0 ? '再涨 ¥' + F.money(gapA) + '（+' + (cnNowM ? (gapA / cnNowM * 100).toFixed(0) : '∞') + '%）'
                      : '已经超出 ¥' + F.money(-gapA)) +
          '</div></div>';
      }
      body += '</div>';

      /* ---- 纯购买力对照 ---- */
      body += '<div class="note" style="margin:14px 0">' +
        '<b>只换算钱的话是这样：</b>中国的 ' + rA.cur + F.wan(rA.tc, 'CN') + ' 按购买力平价 ≈ 西班牙 €' +
        F.money(x.pppOnly.aTCinB) + '；西班牙的 €' + F.wan(rB.tc, 'ES') + ' ≈ 中国 ¥' + F.money(x.pppOnly.bTCinA) + '。<br>' +
        '但上面的等效薪资和这个数字不一样 —— 差额全部来自工时、假期、环境、成长、风险和两国不同的薪资期望曲线。' +
        '下面这张表把差额拆开给你看。</div>';

      /* ---- 差异归因 ---- */
      var at = x.attribution;
      body += '<h3 class="dual-h3">差异归因：中国 ÷ 西班牙 = ' + at.ratio.toFixed(3) +
        '　<span style="font-weight:400;color:var(--text-faint);font-size:12px">（&gt;1 表示中国这份更划算）</span></h3>';
      var maxLog = 0;
      at.rows.forEach(function (r) { maxLog = Math.max(maxLog, Math.abs(Math.log(r.v || 1))); });
      body += '<div class="attr-list">' + at.rows.map(function (r) {
        var lg = Math.log(r.v || 1);
        var w = maxLog > 0 ? Math.abs(lg) / maxLog * 46 : 0;
        var pos = lg >= 0;
        function fmtSide(v) {
          if (r.unit === '¥/天' || r.unit === '¥/h') return '¥' + F.money(v);
          if (r.unit === 'h/天') return v.toFixed(2) + ' h';
          return v.toFixed(3);
        }
        return '<div class="attr-row">' +
          '<div class="attr-l">' + esc(r.label) + '</div>' +
          '<div class="attr-vals">🇨🇳 ' + fmtSide(r.a) + '　🇪🇸 ' + fmtSide(r.b) + '</div>' +
          '<div class="tornado-track"><div class="axis"></div>' +
            '<i class="' + (pos ? 'pos' : 'neg') + '" style="width:' + w.toFixed(1) + '%;' +
            (pos ? 'left:50%' : 'right:50%') + '"></i></div>' +
          '<div class="attr-x">×' + r.v.toFixed(3) + '</div></div>';
      }).join('') + '</div>';
      body += '<div style="font-size:11.5px;color:var(--text-faint);margin-top:8px">' +
        '七个因子连乘 = ' + at.product.toFixed(3) + '，等于左边的得分比。绿色 = 这一项中国占优，红色 = 西班牙占优。</div>';

      /* ---- 并排明细 ---- */
      body += '<h3 class="dual-h3">逐项明细</h3><div class="tbl-scroll"><table class="tbl"><thead><tr>' +
        '<th>指标</th><th class="num">🇨🇳 中国</th><th class="num">🇪🇸 西班牙</th><th class="num">差异</th>' +
        '</tr></thead><tbody>';
      var lines = [
        ['得分（各自本地刻度）', rA.score.toFixed(2), rB.score.toFixed(2), null],
        ['年总包（本币）', rA.cur + F.money(rA.tc), rB.cur + F.money(rB.tc), null],
        ['PPP 标准化年包', '¥' + F.money(rA.pppTC), '¥' + F.money(rB.pppTC), rA.pppTC / rB.pppTC],
        ['年实际工作天数', rA.workDays.toFixed(0) + ' 天', rB.workDays.toFixed(0) + ' 天', rB.workDays / rA.workDays],
        ['有效工时 / 天', rA.effHours.toFixed(2) + ' h', rB.effHours.toFixed(2) + ' h', rB.effHours / rA.effHours],
        ['年被占用工时', F.money(rA.annualHours) + ' h', F.money(rB.annualHours) + ' h', rB.annualHours / rA.annualHours],
        ['PPP 名义时薪', '¥' + F.money(rA.pppRawHourly), '¥' + F.money(rB.pppRawHourly), rA.pppRawHourly / rB.pppRawHourly],
        ['PPP 实感时薪', '¥' + F.money(rA.pppAdjHourly), '¥' + F.money(rB.pppAdjHourly), rA.pppAdjHourly / rB.pppAdjHourly],
        ['本国基准时薪', rA.cur + rA.baseHourly + '（≈¥' + (rA.baseHourly * rA.pppMul).toFixed(0) + '）',
                        rB.cur + rB.baseHourly + '（≈¥' + (rB.baseHourly * rB.pppMul).toFixed(0) + '）', null],
        ['期望系数', rA.expectFactor.toFixed(3), rB.expectFactor.toFixed(3), null],
        ['环境系数', rA.envFactor.toFixed(3), rB.envFactor.toFixed(3), null],
        ['成长系数', rA.growthFactor.toFixed(3), rB.growthFactor.toFixed(3), null],
        ['风险系数', rA.riskFactor.toFixed(3), rB.riskFactor.toFixed(3), null],
        ['有效税负率', (rA.effectiveTaxRate * 100).toFixed(1) + '%', (rB.effectiveTaxRate * 100).toFixed(1) + '%', null]
      ];
      lines.forEach(function (l) {
        body += '<tr><td>' + esc(l[0]) + '</td><td class="num">' + esc(l[1]) + '</td><td class="num">' +
          esc(l[2]) + '</td><td class="num" style="color:var(--text-faint)">' +
          (l[3] ? '×' + l[3].toFixed(2) : '—') + '</td></tr>';
      });
      body += '</tbody></table></div>';
    } else {
      body = '<div class="offer-empty">两边都设定好之后，这里会出现等效薪资求解和差异归因。</div>';
    }

    box.innerHTML = head + slots + body;

    $$('[data-dualset]', box).forEach(function (b) {
      b.onclick = function () {
        var code = b.dataset.dualset;
        if (state.country !== code) {
          alert('当前编辑区是「' + C().name + '」。\n请先在顶部把国家切到「' + cc(code).name + '」，填好之后再存。');
          return;
        }
        dual[code] = JSON.parse(JSON.stringify(state));
        save(); renderDual(); toast('已存入 ' + cc(code).name);
      };
    });
    $$('[data-dualload]', box).forEach(function (b) {
      b.onclick = function () { loadState(dual[b.dataset.dualload]); toast('已载入编辑区'); };
    });
    $$('[data-dualclear]', box).forEach(function (b) {
      b.onclick = function () { dual[b.dataset.dualclear] = null; save(); renderDual(); };
    });
  }

  /* =====================================================================
   * 模型说明
   * ===================================================================== */
  function renderAbout() {
    var Cn = cc('CN'), Es = cc('ES');
    function table(title, rows, cols) {
      return '<h4 style="margin:18px 0 8px;font-size:13px">' + title + '</h4>' +
        '<div class="tbl-scroll"><table class="tbl"><thead><tr>' +
        cols.map(function (c) { return '<th' + (c.num ? ' class="num"' : '') + '>' + esc(c.t) + '</th>'; }).join('') +
        '</tr></thead><tbody>' + rows.map(function (r) {
          return '<tr>' + r.map(function (v, i) {
            return '<td' + (cols[i].num ? ' class="num"' : '') + '>' + v + '</td>';
          }).join('') + '</tr>';
        }).join('') + '</tbody></table></div>';
    }

    var h = '';
    h += '<div class="note info">得分 <b>1.00</b> 的含义：这份工作<b>刚好对得起</b>你在<b>所在国</b>的年限、学历与时间投入。' +
      '大于 1 是赚，小于 1 是亏，多数人会落在 0.7 ~ 1.3 之间。<br>' +
      '每个国家有自己的基准时薪（中国 ¥60/h，西班牙 €12/h），所以<b>得分只能在同一国家内横向比</b>。' +
      '要跨国比，请用 PPP 标准化指标，或直接用「🌍 中西对照」标签页。</div>';

    h += '<h3 style="font-size:14px;margin:20px 0 6px">完整公式</h3>' +
      '<div class="formula card" style="padding:16px"><div style="text-align:center;font-size:13px;line-height:2">' +
      '<span class="up">标准化日薪 × 环境系数 × 成长系数</span><br>' +
      '<span style="display:inline-block;border-top:1px solid var(--border);padding-top:6px;min-width:340px">' +
      '<span class="down">基准时薪 × 有效工时 × 期望系数 × 风险系数</span></span></div></div>';

    h += '<h3 style="font-size:14px;margin:22px 0 6px">七个计算步骤</h3>' +
      '<ol style="font-size:12.5px;line-height:1.9;padding-left:20px;color:var(--text-dim)">' +
      '<li><b>年实际工作天数</b> = 52 × 每周天数 − (年假×请假难度 + 法定假日 + 病假×0.6)</li>' +
      '<li><b>年总包 TC</b> = 基本薪资 + 奖金 + 补贴×12 + 股票面值×兑现折价 + 公司公积金（中国）；可切换税后口径</li>' +
      '<li><b>标准化日薪</b> = TC ÷ 年实际工作天数</li>' +
      '<li><b>有效工时</b> = min(工时,8) + 加班×补偿系数 + 通勤×办公室占比×舒适度 + On-call折算 − 0.5×摸鱼</li>' +
      '<li><b>环境系数</b> = 城市生活成本 × (人文环境加权平均 + 福利加分)</li>' +
      '<li><b>成长系数</b> = 七项技术成长维度的加权平均</li>' +
      '<li><b>期望 / 风险系数</b>（分母）= 学历×年限赛道 / 公司基础风险 + 风险加点 + 年龄风险</li></ol>';

    h += '<h3 style="font-size:14px;margin:22px 0 6px">🌍 两国的结构性差异</h3>';
    h += table('国家参数总览', [
      ['基准时薪（应届 / 中位城市 / 中位环境）', '¥60 / 小时', '€12 / 小时'],
      ['PPP 换算因子（本币 / 国际元）', '4.19', '0.62'],
      ['1 单位本币 ≈ 中国购买力', '¥1.00', '¥6.76'],
      ['薪资曲线：应届 → 5~8 年', '×2.95', '×2.08　<span style="color:var(--warn)">平得多</span>'],
      ['薪资曲线：应届 → 15 年+', '×3.90', '×2.80'],
      ['个人社保', '五险 10.5% + 公积金 5~12%', '6.48%（4.70+1.55+0.10+0.13 MEI）'],
      ['社保基数上限', '各地不同，一线约 ¥36,000/月', '€4.909,50 / 月'],
      ['所得税', '7 级超额累进，起征 ¥60,000/年', 'IRPF 19/24/30/37/45/47%'],
      ['法定假日', '13 天', '14 días festivos'],
      ['常见年假', '5 ~ 15 天', '22 ~ 25 días laborables（法定最低 22）'],
      ['年龄风险强度', '32 岁拐点，35+ 明显惩罚', '只有中国的 1/3 左右，45 岁后才有'],
      ['特色坑位', '外包 / 人力外派 / 驻场', 'Cárnica / consultora（body shopping）']
    ], [{ t: '维度' }, { t: '🇨🇳 中国大陆' }, { t: '🇪🇸 西班牙' }]);

    h += table('工作年限 → 基准薪资倍数',
      Cn.years.map(function (y, i) {
        return [esc(y.label.replace('应届 / 1 年以内', '应届 / Junior')), y.v.toFixed(2), Es.years[i].v.toFixed(2)];
      }),
      [{ t: '年限' }, { t: '🇨🇳 倍数', num: true }, { t: '🇪🇸 倍数', num: true }]);

    h += table('公司类型 🇨🇳', Cn.companyTypes.map(function (c) {
      return [esc(c.label), c.track.toFixed(2), c.risk.toFixed(2), c.market.toFixed(2), esc(c.hint)];
    }), [{ t: '类型' }, { t: '涨薪期望', num: true }, { t: '基础风险', num: true }, { t: '年龄惩罚调节', num: true }, { t: '说明' }]);

    h += table('公司类型 🇪🇸', Es.companyTypes.map(function (c) {
      return [esc(c.label), c.track.toFixed(2), c.risk.toFixed(2), c.market.toFixed(2), esc(c.hint)];
    }), [{ t: '类型' }, { t: '涨薪期望', num: true }, { t: '基础风险', num: true }, { t: '年龄惩罚调节', num: true }, { t: '说明' }]);

    h += table('城市生活成本系数',
      Cn.cities.map(function (c, i) {
        var e = Es.cities[i];
        return [esc(c.label), c.v.toFixed(2), e ? esc(e.label) : '—', e ? e.v.toFixed(2) : '—'];
      }).concat(Es.cities.slice(Cn.cities.length).map(function (e) {
        return ['—', '—', esc(e.label), e.v.toFixed(2)];
      })),
      [{ t: '🇨🇳 城市' }, { t: '系数', num: true }, { t: '🇪🇸 城市' }, { t: '系数', num: true }]);

    h += table('技术栈市场价值系数（全球通用）',
      D.GROWTH_DIMS[0].options.map(function (o) { return [esc(o.label), o.v.toFixed(2), esc(o.hint || '')]; }),
      [{ t: '技术方向' }, { t: '系数', num: true }, { t: '说明' }]);

    h += table('股票 / 期权兑现折价',
      D.STOCK_DISCOUNT.map(function (o) { return [esc(o.label), o.v.toFixed(2), esc(o.hint)]; }),
      [{ t: '类型' }, { t: '折价系数', num: true }, { t: '说明' }]);

    h += table('评级分档', D.RATINGS.map(function (r, i) {
      var lo = i === 0 ? '0' : D.RATINGS[i - 1].max.toFixed(2);
      var hi = r.max === Infinity ? '∞' : r.max.toFixed(2);
      return ['<span style="color:' + r.color + '">' + r.emoji + ' ' + esc(r.label) + '</span>', lo + ' ~ ' + hi, esc(r.desc)];
    }), [{ t: '评级' }, { t: '区间', num: true }, { t: '含义' }]);

    h += '<h3 style="font-size:14px;margin:22px 0 6px">标定依据与数据来源</h3>';
    h += '<div style="font-size:12px;line-height:1.9;color:var(--text-dim)"><b>🇨🇳 中国</b><ul style="padding-left:20px;margin:4px 0">' +
      Cn.sources.map(function (s) { return '<li>' + esc(s) + '</li>'; }).join('') + '</ul>' +
      '<b>🇪🇸 西班牙</b><ul style="padding-left:20px;margin:4px 0">' +
      Es.sources.map(function (s) { return '<li>' + esc(s) + '</li>'; }).join('') + '</ul>' +
      '<b>通用</b><ul style="padding-left:20px;margin:4px 0">' +
      '<li>环境维度权重：Stack Overflow 2025 开发者调查，「自主权与信任」排满意度第 1 位，高于薪酬</li>' +
      '<li>PPP 换算因子沿用 zippland/worth-calculator 的数据源</li></ul></div>';

    h += '<div class="note" style="margin-top:18px"><b>请这样看待这个分数：</b>它是一个主观权重模型，不是统计拟合。' +
      '所有系数都写在 <code>assets/data.js</code> 里，改一个文件就能把它调成你自己的价值观。' +
      '它回答的是「当下这一年划不划算」，不回答「五年后值不值」—— 期权、平台背书、移民身份、' +
      '语言与文化适应成本这些长期变量，模型给不了答案。</div>';

    $('#about').innerHTML = h;
  }

  /* =====================================================================
   * 导出
   * ===================================================================== */
  function textReport() {
    var r = M.compute(state), code = r.country.key, L = [];
    L.push('===== 程序员版工作性价比报告 · ' + r.country.flag + ' ' + r.country.name + ' =====', '');
    L.push('得分：' + r.score.toFixed(2) + '   评级：' + r.rating.emoji + ' ' + r.rating.label);
    L.push(r.rating.desc, '');
    L.push('--- 关键指标 ---');
    L.push('年总包（' + (state.afterTax ? '税后' : '税前') + '）：' + r.cur + F.wan(r.tc, code));
    if (r.stockRaw > 0) L.push('  其中股票面值 ' + r.cur + F.wan(r.stockRaw, code) + ' → 折价后 ' + r.cur + F.wan(r.stockValue, code) + '（×' + r.stockDiscount + '）');
    if (state.afterTax) L.push('  ' + r.country.labels.taxName + ' ' + r.cur + F.money(r.tax) +
      '，个人社保 ' + r.cur + F.money(r.personalSocial) + '，有效税负 ' + (r.effectiveTaxRate * 100).toFixed(1) + '%');
    L.push('年实际工作天数：' + r.workDays.toFixed(1) + ' 天');
    L.push('有效日薪：' + r.cur + F.money(r.dailySalary));
    L.push('有效工时：' + r.effHours.toFixed(2) + ' 小时/天');
    L.push('名义时薪：' + r.cur + F.money(r.rawHourly));
    L.push('实感时薪：' + r.cur + F.money(r.adjHourly) + '（已折算环境与成长）');
    L.push('市场应给时薪：' + r.cur + F.money(r.deservedHourly));
    L.push('PPP 标准化实感时薪：¥' + F.money(r.pppAdjHourly) + '（折算成中国购买力，跨国可比）');
    L.push('年被这份工作占用：' + F.money(r.annualHours) + ' 小时', '');
    L.push('--- 五维评分 ---');
    D.RADAR_DIMS.forEach(function (d) { L.push('  ' + d.label + '：' + Math.round(r.radar[d.key]) + ' / 100'); });
    L.push('', '--- 系数拆解 ---');
    L.push('环境系数 ' + r.envFactor.toFixed(3) + '（城市 ' + r.cityFactor.toFixed(2) + ' × 人文 ' +
      r.envHuman.toFixed(3) + ' + 福利 ' + r.perkBonus.toFixed(3) + '）');
    L.push('成长系数 ' + r.growthFactor.toFixed(3));
    L.push('期望系数 ' + r.expectFactor.toFixed(3) + '（学历 ' + r.eduFactor.toFixed(2) + ' × 年限赛道 ' + r.yearFactor.toFixed(2) + '）');
    L.push('风险系数 ' + r.riskFactor.toFixed(3), '');
    L.push('--- 最值得改善的项 ---');
    M.diagnose(state, 5).forEach(function (d, i) {
      L.push((i + 1) + '. [' + d.group + '] ' + d.label + (d.target ? ' → ' + d.target : '') +
        '   ' + F.pct(d.pct) + '（得分 ' + d.to.toFixed(2) + '）');
    });
    if (dual.CN && dual.ES) {
      var x = M.crossCompare(dual.CN, dual.ES);
      L.push('', '--- 🌍 中西对照 ---');
      L.push('中国：' + x.a.score.toFixed(2) + ' 分，¥' + F.money(x.a.tc) + '，有效工时 ' + x.a.effHours.toFixed(2) + ' h/天');
      L.push('西班牙：' + x.b.score.toFixed(2) + ' 分，€' + F.money(x.b.tc) + '，有效工时 ' + x.b.effHours.toFixed(2) + ' h/天');
      if (x.bNeedsToMatchA) L.push('西班牙要拿 €' + F.money(x.bNeedsToMatchA.annual) + ' bruto 才能追平中国');
      if (x.aNeedsToMatchB) L.push('中国要拿 ¥' + F.money(x.aNeedsToMatchB.value) + '/月（年约 ¥' +
        F.money(x.aNeedsToMatchB.annual) + '）才能追平西班牙');
      L.push('纯购买力换算：中国 ¥' + F.money(x.a.tc) + ' ≈ €' + F.money(x.pppOnly.aTCinB) +
        '；西班牙 €' + F.money(x.b.tc) + ' ≈ ¥' + F.money(x.pppOnly.bTCinA));
    }
    L.push('', '生成于 ' + new Date().toLocaleString('zh-CN'));
    return L.join('\n');
  }

  function download(name, content, type) {
    var blob = new Blob([content], { type: type || 'text/plain;charset=utf-8' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = name;
    document.body.appendChild(a); a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 0);
  }
  function toast(msg) {
    var t = $('#toast');
    t.textContent = msg; t.style.opacity = '1';
    clearTimeout(t._h); t._h = setTimeout(function () { t.style.opacity = '0'; }, 1900);
  }

  /* =====================================================================
   * 国家切换 / 载入
   * ===================================================================== */
  function renderCountryBar() {
    $('#countryBar').innerHTML = D.COUNTRY_LIST.map(function (c) {
      return '<button class="cbtn' + (state.country === c.key ? ' on' : '') + '" data-country="' + c.key + '">' +
        c.flag + ' ' + esc(c.name.split(' · ')[0]) + '</button>';
    }).join('');
    $$('[data-country]').forEach(function (b) {
      b.onclick = function () {
        if (state.country === b.dataset.country) return;
        state = M.switchCountry(state, b.dataset.country);
        rebuild();
        toast('已切换到 ' + cc(state.country).name + '，薪资与国家相关项已重置为当地默认值');
      };
    });
  }

  function loadState(s) {
    if (!s) return;
    state = M.assign(M.defaultState(s.country || 'CN'), s);
    rebuild();
  }

  function rebuild() {
    renderCountryBar();
    renderForm();
    renderPresets();
    update();
    renderOffers();
    renderDual();
  }

  function renderPresets() {
    var list = PRESETS[state.country] || [];
    $('#presets').innerHTML = '<span class="pl">' + cc(state.country).flag + ' 快速填入：</span>' +
      list.map(function (p, i) { return '<button class="preset-btn" data-preset="' + i + '">' + esc(p.name) + '</button>'; }).join('');
    $$('[data-preset]').forEach(function (b) {
      b.onclick = function () {
        var p = list[+b.dataset.preset];
        state = M.assign(M.defaultState(state.country), p.patch);
        state.country = list === PRESETS.CN ? 'CN' : 'ES';
        syncFormFromState(); update();
        toast('已填入预设：' + p.name);
      };
    });
  }

  function update() {
    var r = M.compute(state);
    renderScore(r); renderRadar(r); renderKV(r); renderFormula(r);
    renderDiag(); renderSensitivity(); updateDeps(); save();
  }

  /* =====================================================================
   * 事件
   * ===================================================================== */
  function bind() {
    document.addEventListener('click', function (e) {
      var head = e.target.closest ? e.target.closest('.section-head') : null;
      if (head && head.parentElement.classList.contains('section')) {
        var sec = head.parentElement;
        sec.classList.toggle('open');
        var map = {};
        try { map = JSON.parse(localStorage.getItem(LS_OPEN) || '{}'); } catch (err) {}
        map[sec.dataset.sec] = sec.classList.contains('open');
        try { localStorage.setItem(LS_OPEN, JSON.stringify(map)); } catch (err) {}
      }
    });

    function onInput(e) {
      var el = e.target;
      if (el.dataset.key) {
        var k = el.dataset.key;
        if (el.type === 'checkbox') state[k] = el.checked;
        else if (el.type === 'number') state[k] = el.value === '' ? 0 : parseFloat(el.value);
        else state[k] = parseInt(el.value, 10);
        update();
      } else if (el.dataset.checkkey) {
        var key = el.dataset.checkkey;
        var arr = Array.isArray(state[key]) ? state[key].slice() : [];
        var i = arr.indexOf(el.value);
        if (el.checked && i < 0) arr.push(el.value);
        if (!el.checked && i >= 0) arr.splice(i, 1);
        state[key] = arr; update();
      }
    }
    $('#form').addEventListener('input', onInput);
    $('#form').addEventListener('change', onInput);

    $$('.tab').forEach(function (t) {
      t.onclick = function () {
        $$('.tab').forEach(function (x) { x.classList.remove('on'); });
        $$('.tab-body').forEach(function (x) { x.classList.remove('on'); });
        t.classList.add('on');
        $('#' + t.dataset.tab).classList.add('on');
        if (t.dataset.tab === 'offers') renderOffers();
        if (t.dataset.tab === 'dual') renderDual();
      };
    });

    $('#btnReset').onclick = function () {
      if (!confirm('确定要恢复默认值吗？当前填写的内容会丢失。')) return;
      state = M.defaultState(state.country); rebuild(); toast('已重置');
    };

    $('#btnSaveOffer').onclick = function () {
      if (offers.length >= 6) { toast('最多保存 6 份，先删掉一份'); return; }
      var r = M.compute(state);
      var def = cc(state.country).flag + ' ' + r.company.label.split('（')[0] + ' ' + r.score.toFixed(2);
      var name = prompt('给这份 Offer 起个名字：', def);
      if (name === null) return;
      offers.push({ name: name || def, state: JSON.parse(JSON.stringify(state)) });
      save(); renderOffers();
      showTab('offers');
      toast('已保存，共 ' + offers.length + ' 份');
    };

    $('#btnDual').onclick = function () {
      dual[state.country] = JSON.parse(JSON.stringify(state));
      save(); renderDual(); showTab('dual');
      toast('已存入「' + cc(state.country).name + '」槽位');
    };

    $('#btnCopy').onclick = function () {
      var txt = textReport();
      if (navigator.clipboard) {
        navigator.clipboard.writeText(txt).then(function () { toast('报告已复制到剪贴板'); },
          function () { download('性价比报告.txt', txt); });
      } else download('性价比报告.txt', txt);
    };

    $('#btnJson').onclick = function () {
      download('worth-profile.json', JSON.stringify({ state: state, offers: offers, dual: dual }, null, 2), 'application/json');
      toast('已导出 JSON');
    };
    $('#btnImport').onclick = function () { $('#fileImport').click(); };
    $('#fileImport').onchange = function (e) {
      var f = e.target.files[0]; if (!f) return;
      var fr = new FileReader();
      fr.onload = function () {
        try {
          var o = JSON.parse(fr.result);
          if (o.state) state = M.assign(M.defaultState(o.state.country || 'CN'), o.state);
          if (Array.isArray(o.offers)) offers = o.offers;
          if (o.dual) dual = M.assign({ CN: null, ES: null }, o.dual);
          rebuild(); toast('已导入');
        } catch (err) { alert('文件解析失败：' + err.message); }
      };
      fr.readAsText(f); e.target.value = '';
    };

    $('#btnShare').onclick = function () {
      var url = location.origin + location.pathname + '#s=' + encodeURIComponent(encodeState(state));
      if (navigator.clipboard) {
        navigator.clipboard.writeText(url).then(function () { toast('分享链接已复制'); },
          function () { prompt('复制这个链接：', url); });
      } else prompt('复制这个链接：', url);
      location.hash = 's=' + encodeURIComponent(encodeState(state));
    };

    $('#btnPrint').onclick = function () {
      $$('.section').forEach(function (s) { s.classList.add('open'); });
      setTimeout(function () { window.print(); }, 60);
    };

    $('#btnTheme').onclick = function () {
      var t = document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
      document.documentElement.setAttribute('data-theme', t);
      try { localStorage.setItem(LS_THEME, t); } catch (e) {}
    };
  }

  function showTab(id) {
    $$('.tab').forEach(function (x) { x.classList.remove('on'); });
    $$('.tab-body').forEach(function (x) { x.classList.remove('on'); });
    var t = $('[data-tab="' + id + '"]');
    if (t) t.classList.add('on');
    var b = $('#' + id); if (b) b.classList.add('on');
  }

  /* =====================================================================
   * 启动
   * ===================================================================== */
  function init() {
    try { var th = localStorage.getItem(LS_THEME); if (th) document.documentElement.setAttribute('data-theme', th); } catch (e) {}
    load();
    var fromHash = readHash();
    if (fromHash) state = M.assign(M.defaultState(fromHash.country || 'CN'), fromHash);
    renderAbout();
    bind();
    rebuild();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();

})();
