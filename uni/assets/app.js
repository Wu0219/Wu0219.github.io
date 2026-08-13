/* ===========================================================================
 * app.js —— 界面层
 * 只负责渲染和交互，所有计算都在 model.js 里，所有系数都在 data.js 里。
 * =========================================================================== */
(function () {
  'use strict';
  var D = window.UNI_DATA, M = window.UNI_MODEL, QR = window.UNI_QR;

  /* 分享图二维码指向的地址。换域名时改这一行即可。 */
  var SHARE_URL = 'https://yuhangwu.com/uni/';

  var LS_KEY = 'uni-calc-state';
  var state = load() || M.defaultState('CN');
  var closed = {};

  function load() {
    try { var s = JSON.parse(localStorage.getItem(LS_KEY)); return s && s.country ? s : null; }
    catch (e) { return null; }
  }
  function save() { try { localStorage.setItem(LS_KEY, JSON.stringify(state)); } catch (e) {} }

  var $ = function (id) { return document.getElementById(id); };
  function el(tag, cls, html) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  }
  function toast(msg) {
    var t = $('toast'); t.textContent = msg; t.style.opacity = '1';
    clearTimeout(toast._t); toast._t = setTimeout(function () { t.style.opacity = '0'; }, 1900);
  }

  /* ---------------- 表单 ---------------- */
  function field(f) {
    var wrap = el('div', 'f' + (f.type === 'text' || f.wide ? ' wide' : ''));
    wrap.appendChild(el('label', null, esc(f.label)));

    var input;
    if (f.type === 'select') {
      input = el('select');
      f.options.forEach(function (o) {
        var op = el('option', null, esc(o.label));
        op.value = o.value;
        input.appendChild(op);
      });
      input.value = state[f.key];
      input.onchange = function () { state[f.key] = parseInt(this.value, 10); refresh(); };
    } else if (f.type === 'number') {
      var box = el('div', 'unit');
      input = el('input'); input.type = 'number';
      if (f.min != null) input.min = f.min;
      if (f.max != null) input.max = f.max;
      if (f.step != null) input.step = f.step;
      input.value = state[f.key];
      input.oninput = function () { state[f.key] = parseFloat(this.value); refresh(); };
      box.appendChild(input);
      if (f.unit) box.appendChild(el('span', null, esc(f.unit)));
      wrap.appendChild(box);
    } else {
      input = el('input'); input.type = 'text';
      if (f.placeholder) input.placeholder = f.placeholder;
      input.value = state[f.key] || '';
      input.oninput = function () { state[f.key] = this.value; refresh(); };
    }
    if (f.type !== 'number') wrap.appendChild(input);

    // 选项自带的说明，随选择变化
    var hint = f.hint || '';
    if (f.type === 'select') {
      var cur = f.options[state[f.key]];
      if (cur && cur.hint) hint = cur.hint;
    }
    if (hint) wrap.appendChild(el('div', 'hint', esc(hint)));
    return wrap;
  }

  function renderForm() {
    var host = $('form'); host.innerHTML = '';

    // 国家选择单独放在最前面
    var c0 = el('div', 'card'); c0.style.marginBottom = '14px';
    var h0 = el('div', 'section-head');
    h0.style.cursor = 'default';
    h0.innerHTML = '<div class="ico">🌍</div><div><h2>国家 / 地区</h2>' +
                   '<div class="desc">决定币种与当地花销基准，境外结果会按购买力折算成人民币</div></div>';
    c0.appendChild(h0);
    var b0 = el('div', 'section-body');
    var cf = el('div', 'f wide');
    cf.appendChild(el('label', null, '国家 / 地区'));
    var cs = el('select');
    D.COUNTRIES.forEach(function (C) {
      var op = el('option', null, esc(C.label + '（' + C.cur + '）'));
      op.value = C.key; cs.appendChild(op);
    });
    cs.value = state.country;
    cs.onchange = function () { state = M.switchCountry(state, this.value); renderForm(); refresh(); };
    cf.appendChild(cs);
    b0.appendChild(cf); c0.appendChild(b0); host.appendChild(c0);

    D.buildSections(state.country).forEach(function (sec) {
      var card = el('div', 'card'); card.style.marginBottom = '14px';
      var head = el('div', 'section-head');
      head.innerHTML = '<div class="ico">' + esc(sec.icon) + '</div><div><h2>' + esc(sec.title) +
                       '</h2><div class="desc">' + esc(sec.desc || '') + '</div></div>';
      var body = el('div', 'section-body' + (closed[sec.key] ? ' closed' : ''));
      head.onclick = function () { closed[sec.key] = !closed[sec.key]; body.className = 'section-body' + (closed[sec.key] ? ' closed' : ''); };
      sec.fields.forEach(function (f) { body.appendChild(field(f)); });
      card.appendChild(head); card.appendChild(body); host.appendChild(card);
    });
  }

  /* ---------------- 雷达图 ---------------- */
  function radar(r) {
    /* 颜色从主题变量读，别写死 —— 写死 t1 的深色配色后，
     * 六个浅色主题下标签在白底上只有 2.7:1，看不清。 */
    var cs = getComputedStyle(document.documentElement);
    var cGrid = (cs.getPropertyValue('--bd') || '#26333d').trim();
    var cLbl  = (cs.getPropertyValue('--dim') || '#8fa1ad').trim();
    var cAc   = (cs.getPropertyValue('--ac') || '#35d39a').trim();
    var dims = D.RADAR_DIMS, n = dims.length, cx = 130, cy = 118, R = 82;
    var pts = [], grid = '', axes = '', labels = '';
    for (var g = 1; g <= 4; g++) {
      var p = [];
      for (var i = 0; i < n; i++) {
        var a = -Math.PI / 2 + i * 2 * Math.PI / n;
        p.push((cx + Math.cos(a) * R * g / 4).toFixed(1) + ',' + (cy + Math.sin(a) * R * g / 4).toFixed(1));
      }
      grid += '<polygon points="' + p.join(' ') + '" fill="none" stroke="'+cGrid+'" stroke-width="1"/>';
    }
    for (i = 0; i < n; i++) {
      a = -Math.PI / 2 + i * 2 * Math.PI / n;
      var x = cx + Math.cos(a) * R, y = cy + Math.sin(a) * R;
      axes += '<line x1="' + cx + '" y1="' + cy + '" x2="' + x.toFixed(1) + '" y2="' + y.toFixed(1) + '" stroke="'+cGrid+'"/>';
      var v = Math.max(0, Math.min(100, r.radar[dims[i].key])) / 100;
      pts.push((cx + Math.cos(a) * R * v).toFixed(1) + ',' + (cy + Math.sin(a) * R * v).toFixed(1));
      var lx = cx + Math.cos(a) * (R + 20), ly = cy + Math.sin(a) * (R + 20);
      labels += '<text x="' + lx.toFixed(1) + '" y="' + (ly + 4).toFixed(1) + '" fill="'+cLbl+'" font-size="11.5" text-anchor="middle">' + dims[i].label + '</text>';
    }
    $('radar').innerHTML = '<svg width="260" height="240">' + grid + axes +
      '<polygon points="' + pts.join(' ') + '" fill="'+cAc+'" fill-opacity="0.18" stroke="'+cAc+'" stroke-width="2"/>' +
      labels + '</svg>';
    $('radarLegend').innerHTML = dims.map(function (d) {
      return d.label + ' ' + Math.round(r.radar[d.key]);
    }).join(' · ') + ' &nbsp;(50 = 当地正常)';
  }

  /* ---------------- 结果 ---------------- */
  function refresh() {
    save();
    var r = M.compute(state);
    var f2 = function (n) { return (Math.round(n * 100) / 100).toFixed(2); };
    var money = M.fmt.money;

    $('stageTag').textContent = state.stage === 1 ? '授课型硕士' : '本科生';
    $('scoreNum').textContent = f2(r.score);
    $('scoreNum').style.color = r.rating.color;
    $('scoreRating').innerHTML = r.rating.emoji + ' ' + esc(r.rating.title || r.rating.label) +
      '<span style="color:var(--faint);font-weight:400;font-size:13px"> · ' + esc(r.rating.label) + '</span>';
    $('scoreRating').style.color = r.rating.color;
    $('scoreMark').style.left = Math.max(0, Math.min(100, r.score / 3.0 * 100)) + '%';
    $('scoreDesc').textContent = r.rating.desc;


    radar(r);

    var cur = r.cur, ppp = r.pppMul;
    var kv = [
      ['月支出', cur + money(r.grossMonthly)],
      ['当地基准', cur + money(r.baselineMonthly)],
      ['净支出（扣实习）', cur + money(r.netMonthly)],
      ['成本比', f2(r.costRatio) + (r.costFloored ? ' 触底' : '')]
    ];
    if (r.intern.has) {
      kv.push(['实习月入', cur + money(r.intern.grossIncome)]);
      kv.push(['计入抵扣', cur + money(r.intern.creditedIncome)]);
      kv.push(['周投入（含通勤）', r.intern.weeklyHours + ' 小时']);
      kv.push(['时间惩罚', '×' + f2(r.intern.timePenalty)]);
    }
    if (state.country !== 'CN') {
      kv.push(['净支出（人民币购买力）', '¥' + money(r.netMonthly * ppp)]);
      kv.push(['当地基准（人民币购买力）', '¥' + money(r.baselineMonthly * ppp)]);
    }
    kv.push(['学年总支出（10 个月）', cur + money(r.annualNet)]);
    kv.push(['质量系数合计', f2(r.quality)]);
    $('kv').innerHTML = kv.map(function (x) {
      return '<div class="kv"><div class="k">' + esc(x[0]) + '</div><div class="v">' + esc(x[1]) + '</div></div>';
    }).join('');

    var rows = M.diagnose(state, 8);
    $('diag').innerHTML = rows.length ? rows.map(function (d) {
      return '<div class="diag-row"><span class="diag-g">' + esc(d.group) + '</span>' +
             '<span class="diag-l">' + esc(d.label) +
             (d.target ? '<small>→ ' + esc(d.target) + '</small>' : '') + '</span>' +
             '<span class="diag-d">' + M.fmt.pct(d.pct) + '</span></div>';
    }).join('') : '<div style="padding:14px 0;color:var(--faint);font-size:12.5px">' +
      '每一项都已经是最好的了，没有可改进的空间。</div>';

    refresh._r = r;
  }

  /* ---------------- 分享图 ---------------- */
  /* 金额 → 星级。
   * 星星表示的是「好不好」，不是「数字大不大」：
   * 花钱少 = 星多，赚钱多 = 星多。方向搞反的话，
   * 一个靠实习把开销全覆盖的人会拿到一颗星，那显然是错的。 */
  function starBar(n) {
    n = Math.max(1, Math.min(5, Math.round(n)));
    return '★★★★★'.slice(0, n) + '☆☆☆☆☆'.slice(0, 5 - n);
  }
  /* 星级统一从「0~100 饱和分」换算，和五维用同一把尺子。
   * 饱和曲线的上界够不着 100（最好情况约 91），而满星要求 ≥92，
   * 所以满星实际上拿不到 —— 分数只能无限趋近满分，不会真的封顶。 */
  function starsFromScore(v) {
    return starBar(v < 25 ? 1 : v < 45 ? 2 : v < 63 ? 3 : v < 92 ? 4 : 5);
  }
  function costScore(ratio) {          // 支出：越低越好
    return 100 - M.pivot(Math.max(ratio, 0.05), 0.30, 1.0, 2.20);
  }
  function incomeScore(ratio) {        // 收入：越高越好
    return M.pivot(Math.max(ratio, 0), 0, 0.55, 1.30);
  }

  /* 星星统一表示「好不好」，描述词补上「实际是多还是少」。
   * 只给星星的话，「花得少」和「赚得多」都是满星，看不出差别；
   * 加一个词就能同时读出方向和好坏。 */
  function costWord(ratio) {
    if (!isFinite(ratio) || ratio <= 0) return '几乎不用掏';
    return ratio < 0.55 ? '很省' : ratio < 0.80 ? '偏省' : ratio < 1.15 ? '正常' : ratio < 1.60 ? '偏高' : '很高';
  }
  function incomeWord(ratio) {
    if (!isFinite(ratio) || ratio <= 0) return '暂无';
    return ratio < 0.20 ? '很少' : ratio < 0.45 ? '补贴零用' : ratio < 0.80 ? '覆盖大半' : ratio < 1.15 ? '够养活自己' : '有结余';
  }
  function levelWord(v100) {           // 0~100 的维度分 → 档次词
    return v100 < 30 ? '较差' : v100 < 45 ? '一般' : v100 < 60 ? '还行' : v100 < 78 ? '不错' : '很好';
  }

  function drawShare(hideMoney, withQR) {
    var r = refresh._r;
    /* 固定 3:4 —— 小红书信息流里封面按 3:4 展示，比例不对会比别人矮一截。 */
    var W = 780, H = 1040, dpr = 2;
    var cv = document.createElement('canvas');
    cv.width = W * dpr; cv.height = H * dpr;
    var x = cv.getContext('2d'); x.scale(dpr, dpr);
    var SANS = '"Microsoft YaHei","PingFang SC",-apple-system,sans-serif';
    var MONO = 'Consolas,"SF Mono",monospace';

    /* 配色跟随用户当前选的主题。写死深色的话，选了「俏皮糖果」的用户
     * 生成出来还是一张黑客风的图，九套主题等于白做。 */
    var cs2 = getComputedStyle(document.documentElement);
    function V(name, dflt) { return (cs2.getPropertyValue(name) || dflt).trim() || dflt; }
    var C_BG = V('--bg', '#0f1418'), C_CARD = V('--elev', '#161d23'),
        C_BD = V('--bd', '#26333d'), C_TX = V('--tx', '#dde5ea'),
        C_DIM = V('--dim', '#8fa1ad'), C_FAINT = V('--faint', '#61717c'),
        C_AC = V('--ac', '#35d39a'), C_WARN = V('--warn', '#e0a33d');

    x.fillStyle = C_BG; x.fillRect(0, 0, W, H);
    // 顶部渐变条
    var g = x.createLinearGradient(0, 0, W, 0);
    g.addColorStop(0, '#35d39a'); g.addColorStop(1, '#4aa3e0');
    x.fillStyle = g; x.fillRect(0, 0, W, 6);

    var P = 46, y = 62;
    x.fillStyle = C_TX; x.font = '700 30px ' + SANS;
    x.fillText('大学上得值不值', P, y); y += 30;
    x.fillStyle = C_FAINT; x.font = '400 14px ' + SANS;
    x.fillText('大学生活性价比测评 · 本科 / 授课型硕士', P, y); y += 40;

    // 学校 / 专业
    var school = (state.schoolName || '').trim() || '（未填写学校）';
    var major = (state.majorName || '').trim();
    x.fillStyle = C_TX; x.font = '600 22px ' + SANS;
    x.fillText(school.length > 22 ? school.slice(0, 22) + '…' : school, P, y); y += 26;
    x.fillStyle = C_DIM; x.font = '400 15px ' + SANS;
    var line2 = [major, r.country.label + ' · ' + r.region.label, r.loc.label]
      .filter(Boolean).join('  ·  ');
    x.fillText(line2.length > 40 ? line2.slice(0, 40) + '…' : line2, P, y); y += 34;

    /* 主分数卡：名号是第一视觉层级，分数退到第二。
     * 缩略图里只看得清一个东西，而「1.42」对路人毫无意义 ——
     * 不知道满分多少、是高是低；「标准大学生🙂」才是能被读懂和转发的。 */
    x.fillStyle = C_CARD; roundRect(x, P, y, W - P * 2, 250, 14); x.fill();
    x.strokeStyle = C_BD; x.lineWidth = 1; x.stroke();
    x.textAlign = 'center';
    x.fillStyle = r.rating.color; x.font = '700 52px ' + SANS;
    x.fillText(r.rating.emoji + ' ' + (r.rating.title || r.rating.label), W / 2, y + 88);
    x.font = '700 40px ' + MONO;
    x.fillText((Math.round(r.score * 100) / 100).toFixed(2), W / 2, y + 152);
    x.fillStyle = C_FAINT; x.font = '400 13.5px ' + SANS;
    x.fillText('1.00 = 当地正常水平', W / 2, y + 182);
    // 刻度条
    var bx = P + 40, bw = W - P * 2 - 80, by = y + 210;
    var bg = x.createLinearGradient(bx, 0, bx + bw, 0);
    ['#e0413a', '#e8763a', '#d9a13a', '#4aa3e0', '#2f9e6e', '#7b4fd9', '#c9971f']
      .forEach(function (c, i, a) { bg.addColorStop(i / (a.length - 1), c); });
    x.fillStyle = bg; roundRect(x, bx, by, bw, 6, 3); x.fill();
    var mx = bx + Math.max(0, Math.min(1, r.score / 3)) * bw;
    x.fillStyle = '#fff'; roundRect(x, mx - 2, by - 5, 4, 16, 2); x.fill();
    x.textAlign = 'left'; y += 286;

    // 五维
    x.fillStyle = C_DIM; x.font = '600 14px ' + SANS;
    x.fillText('五维评分（50 = 当地正常）', P, y); y += 16;
    D.RADAR_DIMS.forEach(function (d) {
      var v = Math.max(0, Math.min(100, r.radar[d.key]));
      x.fillStyle = C_FAINT; x.font = '400 13px ' + SANS;
      x.fillText(d.label, P, y + 14);
      x.fillStyle = C_BD; roundRect(x, P + 46, y + 5, W - P * 2 - 100, 11, 5); x.fill();
      x.fillStyle = v >= 50 ? C_AC : C_WARN;
      roundRect(x, P + 46, y + 5, (W - P * 2 - 100) * v / 100, 11, 5); x.fill();
      x.fillStyle = C_DIM; x.font = '600 12px ' + MONO;
      x.textAlign = 'right'; x.fillText(Math.round(v), W - P, y + 15); x.textAlign = 'left';
      y += 31;
    });
    y += 16;

    // 关键数字（可切换成星级，避免把具体收入晒出去）
    var cur = r.cur, facts;
    /* 每格三行：名称 / 星级（永远是高=好）/ 描述词（说明实际是多是少）。
     * 隐藏金额时把星级放中间，显示金额时把金额放中间，两种模式结构一致。 */
    var base = r.baselineMonthly || 1;
    var incRatio = r.intern.has ? r.intern.grossIncome / base : 0;
    if (hideMoney) {
      facts = [
        ['月支出',   starsFromScore(costScore(r.grossMonthly / base)), costWord(r.grossMonthly / base)],
        ['净支出',   starsFromScore(costScore(r.netMonthly / base)),   costWord(r.netMonthly / base)],
        ['实习收入', incRatio > 0 ? starsFromScore(incomeScore(incRatio)) : '—', incomeWord(incRatio)],
        ['住宿条件', starsFromScore(r.radar.dorm),                     levelWord(r.radar.dorm)]
      ];
    } else {
      facts = [
        ['月支出',   cur + M.fmt.money(r.grossMonthly), costWord(r.grossMonthly / base)],
        ['净支出',   cur + M.fmt.money(r.netMonthly),   costWord(r.netMonthly / base)],
        ['实习收入', r.intern.has ? cur + M.fmt.money(r.intern.grossIncome) : '—', incomeWord(incRatio)],
        ['当地基准', cur + M.fmt.money(r.baselineMonthly), '同类学生水平']
      ];
    }
    var fw = (W - P * 2 - 3 * 10) / 4;
    facts.slice(0, 4).forEach(function (ft, i) {
      var fx = P + i * (fw + 10);
      x.fillStyle = C_CARD; roundRect(x, fx, y, fw, 92, 10); x.fill();
      x.strokeStyle = C_BD; x.stroke();
      x.textAlign = 'center';
      x.fillStyle = C_FAINT; x.font = '400 11px ' + SANS;
      x.fillText(ft[0], fx + fw / 2, y + 19);
      var t = String(ft[1]);
      var isStar = t.indexOf('★') >= 0 || t.indexOf('☆') >= 0;
      x.fillStyle = isStar ? C_WARN : C_TX;
      x.font = isStar ? '400 15px ' + SANS : (t.length > 10 ? '600 13px ' : '600 16px ') + MONO;
      x.fillText(t, fx + fw / 2, y + 45);
      x.fillStyle = C_DIM; x.font = '400 11.5px ' + SANS;
      x.fillText(String(ft[2] || ''), fx + fw / 2, y + 74);
      x.textAlign = 'left';
    });
    y += 116;

    /* 最值得改善的一项 —— 这是最容易引发评论区讨论的内容，
     * 之前只在网页上有、图上没有，等于把最好的话题点漏在了外面。 */
    var top1 = M.diagnose(state, 1)[0];
    if (top1) {
      x.fillStyle = C_CARD; roundRect(x, P, y, W - P * 2, 62, 10); x.fill();
      x.strokeStyle = C_BD; x.stroke();
      x.textAlign = 'left';
      x.fillStyle = C_FAINT; x.font = '400 11.5px ' + SANS;
      x.fillText('最值得改善', P + 20, y + 23);
      x.fillStyle = C_TX; x.font = '600 16px ' + SANS;
      var tl = top1.label + (top1.target ? ' → ' + top1.target : '');
      if (tl.length > 26) tl = tl.slice(0, 26) + '…';
      x.fillText(tl, P + 20, y + 46);
      x.fillStyle = C_AC; x.font = '600 15px ' + MONO;
      x.textAlign = 'right';
      x.fillText(M.fmt.pct(top1.pct), W - P - 20, y + 46);
    }

    /* 底部。默认只放品牌搜索词不放二维码 ——
     * 小红书把二维码和明文网址列为违规导流载体（图片会被 OCR 扫），
     * 命中就是笔记不收录甚至账号降权，平台罚的是导流行为不是提到一个名字。
     * 勾选「附上二维码」后才画，适合私下发给朋友或买家的场合。 */
    if (withQR) {
      var qr = QR.make(SHARE_URL, { ecc: 'H' });
      var qs = Math.max(3, Math.floor(140 / (qr.size + 8))), qFull = (qr.size + 8) * qs;
      var qx = W - P - qFull, qy = H - 30 - qFull;
      x.fillStyle = '#fff'; roundRect(x, qx - 6, qy - 6, qFull + 12, qFull + 12, 6); x.fill();
      QR.draw(x, qr, qx, qy, qs, '#000', 'none');
      x.textAlign = 'left';
      x.fillStyle = C_TX; x.font = '700 24px ' + SANS;
      x.fillText('扫码测测你的大学', P, H - 78);
      x.fillStyle = C_DIM; x.font = '400 14px ' + SANS;
      x.fillText('住宿 · 地段 · 校园 · 前景 · 实习 · 花销', P, H - 50);
      x.fillStyle = C_FAINT; x.font = '400 12.5px ' + SANS;
      x.fillText('微信搜「校值」也能找到', P, H - 26);
    } else {
      x.textAlign = 'center';
      x.fillStyle = C_TX; x.font = '700 26px ' + SANS;
      x.fillText('微信搜「校值」测你的大学', W / 2, H - 74);
      x.fillStyle = C_DIM; x.font = '400 14px ' + SANS;
      x.fillText('住宿 · 地段 · 校园 · 前景 · 实习 · 花销，六维打分', W / 2, H - 46);
      x.fillStyle = C_FAINT; x.font = '400 11.5px ' + SANS;
      x.fillText('校值 · 大学生活性价比测评', W / 2, H - 22);
    }

    return cv;
  }
  function roundRect(x, l, t, w, h, r) {
    x.beginPath();
    x.moveTo(l + r, t); x.arcTo(l + w, t, l + w, t + h, r);
    x.arcTo(l + w, t + h, l, t + h, r); x.arcTo(l, t + h, l, t, r);
    x.arcTo(l, t, l + w, t, r); x.closePath();
  }

  /* ---------------- 文本报告 ---------------- */
  function report() {
    var r = refresh._r, f2 = function (n) { return (Math.round(n * 100) / 100).toFixed(2); };
    var L = [];
    L.push('【大学上得值不值】总分 ' + f2(r.score) + ' ' + r.rating.emoji + ' ' + r.rating.label);
    L.push((state.schoolName || '未填写') + (state.majorName ? ' · ' + state.majorName : ''));
    L.push(r.country.label + ' · ' + r.region.label + ' · ' + r.loc.label);
    L.push('');
    L.push('月支出 ' + r.cur + M.fmt.money(r.grossMonthly) + '，当地基准 ' + r.cur + M.fmt.money(r.baselineMonthly));
    if (r.intern.has)
      L.push('实习月入 ' + r.cur + M.fmt.money(r.intern.grossIncome) + '，每周投入 ' + r.intern.weeklyHours + ' 小时（含通勤）');
    L.push('');
    L.push('五维：' + D.RADAR_DIMS.map(function (d) { return d.label + ' ' + Math.round(r.radar[d.key]); }).join('  '));
    L.push('');
    L.push('最值得改善：');
    M.diagnose(state, 3).forEach(function (d, i) {
      L.push('  ' + (i + 1) + '. ' + d.label + (d.target ? ' → ' + d.target : '') + '  ' + M.fmt.pct(d.pct));
    });
    return L.join('\n');
  }

  /* ---------------- 事件 ---------------- */
  /* ---------------- 详细图片报告 ----------------
   * 一张长图，内容等同于一份完整报告。做成图片而不是 PDF，
   * 是因为微信里长按就能存，而打印到 PDF 在手机上基本走不通。 */
  function drawReport(withQR) {
    var r = M.compute(state);
    var f2 = function (n) { return (Math.round(n * 100) / 100).toFixed(2); };
    var money = M.fmt.money, cur = r.cur;
    var W = 900, dpr = 2, P = 56;
    var SANS = '"Microsoft YaHei","PingFang SC",-apple-system,sans-serif';
    var MONO = 'Consolas,"SF Mono",monospace';

    var cs2 = getComputedStyle(document.documentElement);
    function V(k, d) { return (cs2.getPropertyValue(k) || d).trim() || d; }
    var C_BG = V('--bg', '#fbfaf7'), C_CARD = V('--elev', '#fff'), C_BD = V('--bd', '#d8d3c7'),
        C_TX = V('--tx', '#1f1c17'), C_DIM = V('--dim', '#5d574c'), C_FAINT = V('--faint', '#928b7d'),
        C_AC = V('--ac', '#1b3b6f'), C_WARN = V('--warn', '#a3761d');

    var cv = document.createElement('canvas');
    var x = cv.getContext('2d');
    var y = 0, pass = 0;

    function line(txt, size, color, bold, indent) {
      x.fillStyle = color; x.font = (bold ? '600 ' : '400 ') + size + 'px ' + SANS;
      x.textAlign = 'left'; x.fillText(txt, P + (indent || 0), y + size);
      y += size + 9;
    }
    function wrap(txt, size, color, maxW) {
      x.font = '400 ' + size + 'px ' + SANS; x.fillStyle = color; x.textAlign = 'left';
      var buf = '', out = [];
      for (var i = 0; i < txt.length; i++) {
        if (x.measureText(buf + txt[i]).width > maxW) { out.push(buf); buf = ''; }
        buf += txt[i];
      }
      out.push(buf);
      out.forEach(function (l) { x.fillText(l, P, y + size); y += size + 7; });
      y += 4;
    }
    function h2(txt) {
      y += 20;
      x.fillStyle = C_AC; x.fillRect(P, y + 4, 4, 20);
      x.fillStyle = C_AC; x.font = '700 21px ' + SANS; x.textAlign = 'left';
      x.fillText(txt, P + 14, y + 21);
      y += 34;
      x.strokeStyle = C_BD; x.lineWidth = 1;
      x.beginPath(); x.moveTo(P, y); x.lineTo(W - P, y); x.stroke();
      y += 14;
    }
    function rows(list, colW) {
      list.forEach(function (it) {
        x.fillStyle = C_DIM; x.font = '400 15px ' + SANS; x.textAlign = 'left';
        x.fillText(it[0], P, y + 15);
        x.fillStyle = C_TX; x.font = (it[2] ? '600 ' : '400 ') + '15px ' + (it[2] ? MONO : SANS);
        x.fillText(String(it[1]), P + (colW || 260), y + 15);
        y += 28;
        x.strokeStyle = C_BD; x.globalAlpha = .45;
        x.beginPath(); x.moveTo(P, y - 7); x.lineTo(W - P, y - 7); x.stroke();
        x.globalAlpha = 1;
      });
      y += 6;
    }

    function render() {
      x.fillStyle = C_BG; x.fillRect(0, 0, W, cv.height / dpr);
      var g = x.createLinearGradient(0, 0, W, 0);
      g.addColorStop(0, C_AC); g.addColorStop(1, C_WARN);
      x.fillStyle = g; x.fillRect(0, 0, W, 6);
      y = 54;

      /* 抬头 */
      line('校值 · 大学生活性价比测评报告', 30, C_TX, true);
      y += 2;
      line((state.schoolName || '未填写学校') + (state.majorName ? ' · ' + state.majorName : ''), 19, C_TX, true);
      line(r.country.label + ' ' + r.region.label + ' · ' + r.loc.label + ' · ' +
           (state.stage === 1 ? '授课型硕士' : '本科生'), 15, C_FAINT);

      /* 总分 */
      y += 12;
      x.fillStyle = C_CARD; roundRect(x, P, y, W - P * 2, 128, 12); x.fill();
      x.strokeStyle = C_BD; x.lineWidth = 1; x.stroke();
      x.textAlign = 'left';
      x.fillStyle = C_FAINT; x.font = '400 14px ' + SANS;
      x.fillText('综合得分', P + 26, y + 32);
      x.fillStyle = r.rating.color; x.font = '700 52px ' + MONO;
      x.fillText(f2(r.score), P + 26, y + 88);
      x.font = '700 30px ' + SANS;
      x.textAlign = 'right';
      x.fillText(r.rating.emoji + ' ' + r.rating.title, W - P - 26, y + 62);
      x.fillStyle = C_FAINT; x.font = '400 14px ' + SANS;
      x.fillText('1.00 = 所在城市同类学生中位水平', W - P - 26, y + 92);
      x.textAlign = 'left';
      y += 148;
      wrap(r.rating.desc, 15, C_DIM, W - P * 2);

      /* 一、维度评分 —— 雷达图 + 条形 */
      h2('一、维度评分');
      var dims = D.RADAR_DIMS, n = dims.length;
      var cx = P + 148, cy = y + 150, R = 108;
      for (var ring = 1; ring <= 4; ring++) {
        x.beginPath();
        for (var i = 0; i < n; i++) {
          var a = -Math.PI / 2 + i * 2 * Math.PI / n;
          var px = cx + Math.cos(a) * R * ring / 4, py = cy + Math.sin(a) * R * ring / 4;
          i ? x.lineTo(px, py) : x.moveTo(px, py);
        }
        x.closePath(); x.strokeStyle = C_BD; x.lineWidth = 1; x.stroke();
      }
      x.beginPath();
      for (i = 0; i < n; i++) {
        a = -Math.PI / 2 + i * 2 * Math.PI / n;
        x.moveTo(cx, cy); x.lineTo(cx + Math.cos(a) * R, cy + Math.sin(a) * R);
      }
      x.strokeStyle = C_BD; x.stroke();
      x.beginPath();
      for (i = 0; i < n; i++) {
        a = -Math.PI / 2 + i * 2 * Math.PI / n;
        var v = Math.max(0, Math.min(100, r.radar[dims[i].key])) / 100;
        var qx = cx + Math.cos(a) * R * v, qy = cy + Math.sin(a) * R * v;
        i ? x.lineTo(qx, qy) : x.moveTo(qx, qy);
      }
      x.closePath();
      x.globalAlpha = .18; x.fillStyle = C_AC; x.fill(); x.globalAlpha = 1;
      x.strokeStyle = C_AC; x.lineWidth = 2; x.stroke();
      x.fillStyle = C_DIM; x.font = '500 14px ' + SANS; x.textAlign = 'center';
      for (i = 0; i < n; i++) {
        a = -Math.PI / 2 + i * 2 * Math.PI / n;
        x.fillText(dims[i].label, cx + Math.cos(a) * (R + 24), cy + Math.sin(a) * (R + 24) + 5);
      }
      x.textAlign = 'left';

      // 右侧条形
      var bx0 = cx + R + 76, bw0 = W - P - bx0 - 52, by0 = y + 42;
      dims.forEach(function (d) {
        var v = Math.max(0, Math.min(100, r.radar[d.key]));
        x.fillStyle = C_FAINT; x.font = '400 14px ' + SANS;
        x.fillText(d.label, bx0, by0 + 12);
        x.fillStyle = C_BD; roundRect(x, bx0 + 44, by0 + 2, bw0, 12, 6); x.fill();
        x.fillStyle = v >= 50 ? C_AC : C_WARN;
        roundRect(x, bx0 + 44, by0 + 2, Math.max(bw0 * v / 100, 4), 12, 6); x.fill();
        x.fillStyle = C_TX; x.font = '600 14px ' + MONO; x.textAlign = 'right';
        x.fillText(Math.round(v), W - P, by0 + 13); x.textAlign = 'left';
        by0 += 40;
      });
      y = Math.max(cy + R + 44, by0 + 6);
      x.fillStyle = C_FAINT; x.font = '400 13px ' + SANS;
      x.fillText('50 分为所在城市同类学生的中位水平', P, y); y += 20;

      /* 二、支出结构 */
      h2('二、支出结构');
      var sp = [['每月总支出', cur + money(r.grossMonthly), 1],
                ['当地同类学生基准', cur + money(r.baselineMonthly), 1],
                ['实习月收入', r.intern.has ? cur + money(r.intern.grossIncome) : '无', 1],
                ['实际净支出', cur + money(r.netMonthly), 1],
                ['学年净支出（10 个月）', cur + money(r.annualNet), 1],
                ['支出 / 基准', f2(r.costRatio), 1]];
      rows(sp);
      if (state.country !== 'CN') {
        wrap('按购买力平价折算，净支出相当于人民币 ¥' + money(r.pppNetMonthly) +
             ' / 月，当地基准相当于 ¥' + money(r.pppBaselineMonthly) + ' / 月。', 13, C_FAINT, W - P * 2);
      }

      /* 三、实习 */
      var idx = 3;
      if (r.intern.has) {
        h2('三、实习投入产出');
        rows([['含金量', r.intern.quality.label], ['专业对口度', r.intern.relevance.label],
              ['时间分布', r.intern.term.label], ['转正预期', r.intern.convert.label],
              ['每周投入（含通勤）', r.intern.weeklyHours + ' 小时', 1],
              ['时间成本折减', '×' + f2(r.intern.timePenalty), 1],
              ['前景加成', '×' + f2(r.intern.prospectBoost), 1]]);
        idx = 4;
      }

      /* 改善优先级 */
      var CN = ['', '一', '二', '三', '四', '五', '六', '七', '八'];
      h2(CN[idx] + '、改善优先级');
      var dg = M.diagnose(state, 6);
      if (dg.length) {
        dg.forEach(function (d, i) {
          x.fillStyle = C_AC; x.font = '700 15px ' + MONO; x.textAlign = 'left';
          x.fillText(String(i + 1), P, y + 15);
          x.fillStyle = C_TX; x.font = '500 15px ' + SANS;
          x.fillText(d.label + (d.target ? '  →  ' + d.target : ''), P + 26, y + 15);
          x.fillStyle = C_AC; x.font = '600 15px ' + MONO; x.textAlign = 'right';
          x.fillText(M.fmt.pct(d.pct), W - P, y + 15); x.textAlign = 'left';
          y += 30;
        });
        y += 4;
        wrap('仅列出入学后仍可调整的项目，专业与院校层次不在其中。', 13, C_FAINT, W - P * 2);
      } else { wrap('各项均已处于最优档。', 14, C_DIM, W - P * 2); }

      /* 填写明细 */
      h2(CN[idx + 1] + '、填写明细');
      D.buildSections(state.country).forEach(function (sec) {
        var fs = sec.fields.filter(function (f) { return f.type === 'select'; });
        if (!fs.length) return;
        x.fillStyle = C_AC; x.font = '600 15px ' + SANS; x.textAlign = 'left';
        x.fillText(sec.title, P, y + 15); y += 26;
        fs.forEach(function (f) {
          var o = f.options[state[f.key]]; if (!o) return;
          x.fillStyle = C_FAINT; x.font = '400 14px ' + SANS;
          x.fillText(f.label, P + 14, y + 14);
          x.fillStyle = C_TX;
          x.fillText(o.label, P + 300, y + 14);
          y += 25;
        });
        y += 10;
      });

      /* 计算方法 */
      h2(CN[idx + 2] + '、计算方法');
      wrap('综合得分由质量分与成本比之商构成，再由主观感受修正：', 14, C_DIM, W - P * 2);
      x.fillStyle = C_CARD; roundRect(x, P, y, W - P * 2, 46, 8); x.fill();
      x.strokeStyle = C_BD; x.stroke();
      x.fillStyle = C_AC; x.font = '500 16px ' + SANS; x.textAlign = 'center';
      x.fillText('综合得分 = (住宿 × 地段 × 校园 × 前景)^0.75 × 净收益修正 ÷ 成本比^0.75 × 主观感受^0.45',
                 W / 2, y + 29);
      x.textAlign = 'left'; y += 62;
      wrap('成本比为净支出与所在城市同类学生基准花销之比。四组质量指标各由若干条目加权平均得到，' +
           '以当地中位水平为 1.00。指数 0.75 用于抑制多项指标连乘产生的复合放大。', 14, C_DIM, W - P * 2);
      wrap('跨国比较采用世界银行国际比较项目（ICP）的居民消费购买力平价换算因子：', 14, C_DIM, W - P * 2);
      x.fillStyle = C_CARD; roundRect(x, P, y, W - P * 2, 44, 8); x.fill();
      x.strokeStyle = C_BD; x.stroke();
      x.fillStyle = C_AC; x.font = '500 16px ' + SANS; x.textAlign = 'center';
      x.fillText('人民币等值 = 当地金额 ÷ PPP(当地) × PPP(中国)', W / 2, y + 28);
      x.textAlign = 'left'; y += 60;
      wrap('本报告使用 PPP(中国) = ' + D.GLOBAL.PPP_ANCHOR + '，当前所在地 PPP = ' +
           r.country.ppp + '，折算系数 ' + f2(r.pppMul) + '。', 14, C_DIM, W - P * 2);

      /* 参考来源 */
      h2(CN[idx + 3] + '、参考来源');
      [['购买力平价换算因子',
        '世界银行 国际比较项目（ICP）2021 轮，指标 PA.NUS.PRVT.PP：居民及为居民服务的非营利机构最终消费支出购买力平价（本币／国际元）  data.worldbank.org'],
       ['英国生活费基准',
        'Save the Student《National Student Money Survey 2025》：英国在校学生月度总支出与房租中位数  savethestudent.org'],
       ['西班牙及欧洲房租基准',
        'HousingAnywhere 房租指数 2025 年第三季度：主要城市学生单间月租均价  housinganywhere.com']
      ].forEach(function (it) {
        x.fillStyle = C_TX; x.font = '600 14px ' + SANS; x.textAlign = 'left';
        x.fillText(it[0], P, y + 14); y += 22;
        wrap(it[1], 13, C_FAINT, W - P * 2);
        y += 2;
      });
      wrap('上述来源用于设定各地区的生活费与住宿基准，以及跨国购买力折算。' +
           '各评分条目的权重与评级区间由本工具设定，不来自上述来源。', 13, C_FAINT, W - P * 2);

      y += 22;
      x.strokeStyle = C_BD; x.beginPath(); x.moveTo(P, y); x.lineTo(W - P, y); x.stroke();
      y += 24;
      if (withQR) {
        var qr2 = QR.make(SHARE_URL, { ecc: 'H' });
        var qs2 = Math.max(4, Math.floor(210 / (qr2.size + 8))), qF2 = (qr2.size + 8) * qs2;
        x.fillStyle = '#fff'; roundRect(x, W / 2 - qF2 / 2 - 6, y - 6, qF2 + 12, qF2 + 12, 6); x.fill();
        QR.draw(x, qr2, W / 2 - qF2 / 2, y, qs2, '#000', 'none');
        y += qF2 + 20;
      }
      x.fillStyle = C_FAINT; x.font = '400 14px ' + SANS; x.textAlign = 'center';
      x.fillText('校值 · 大学生活性价比测评 · 微信搜「校值」', W / 2, y);
      x.textAlign = 'left';
      y += 40;
    }

    // 第一遍量高度，第二遍正式画
    cv.width = W * dpr; cv.height = 6000 * dpr;
    x.scale(dpr, dpr); render();
    var finalH = y;
    cv.width = W * dpr; cv.height = finalH * dpr;
    x = cv.getContext('2d'); x.scale(dpr, dpr);
    render();
    return cv;
  }

  function renderShare() {
    var hide = $('hideMoney').checked, qr = $('withQR').checked;
    try {
      localStorage.setItem('uni-hide-money', hide ? '1' : '0');
      localStorage.setItem('uni-with-qr', qr ? '1' : '0');
    } catch (e) {}
    var url;
    try { url = drawShare(hide, qr).toDataURL('image/png'); }
    catch (e) { toast('生成失败：' + e.message); return null; }
    $('shareImg').src = url;
    return url;
  }

  try {
    $('hideMoney').checked = localStorage.getItem('uni-hide-money') === '1';
    $('withQR').checked = localStorage.getItem('uni-with-qr') === '1';
  } catch (e) {}
  // 勾选变化立刻重画当前这张图，不用关掉重来
  var lastKind = 'share';
  function reRender() {
    var u = lastKind === 'report' ? renderReport() : renderShare();
    if (u) bindDownload(u);
  }
  $('hideMoney').onchange = reRender;
  $('withQR').onchange = reRender;

  function renderReport() {
    var qr = $('withQR').checked;
    try { localStorage.setItem('uni-with-qr', qr ? '1' : '0'); } catch (e) {}
    var url;
    try { url = drawReport(qr).toDataURL('image/png'); }
    catch (e) { toast('生成失败：' + e.message); return null; }
    $('shareImg').src = url;
    return url;
  }

  function bindDownload(url) {
    $('btnDl').onclick = function () {
      var a = document.createElement('a');
      a.download = '大学性价比-' + ((state.schoolName || '测评').slice(0, 12)) + '.png';
      a.href = url; a.click();
    };
  }

  $('btnShare').onclick = function () {
    lastKind = 'share';
    $('lblHideMoney').style.display = '';
    var url = renderShare();
    if (!url) return;
    bindDownload(url);
    $('shareMask').classList.add('on');
  };

  $('btnReport').onclick = function () {
    lastKind = 'report';
    // 详细报告是给自己存档的，隐藏金额没有意义，这个开关就不显示
    $('lblHideMoney').style.display = 'none';
    var url = renderReport();
    if (!url) return;
    bindDownload(url);
    $('shareMask').classList.add('on');
  };
  $('btnCloseShare').onclick = function () { $('shareMask').classList.remove('on'); };
  $('shareMask').onclick = function (e) { if (e.target === this) this.classList.remove('on'); };
  $('btnCopy').onclick = function () {
    var t = report();
    if (navigator.clipboard) navigator.clipboard.writeText(t).then(function () { toast('报告已复制'); },
      function () { toast('复制失败，请手动选择'); });
    else toast('浏览器不支持自动复制');
  };
  $('btnReset').onclick = function () {
    if (!confirm('重置所有填写内容？')) return;
    state = M.defaultState(state.country); renderForm(); refresh(); toast('已重置');
  };

  renderForm();
  refresh();
})();
