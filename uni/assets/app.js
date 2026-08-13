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
    var dims = D.RADAR_DIMS, n = dims.length, cx = 130, cy = 118, R = 82;
    var pts = [], grid = '', axes = '', labels = '';
    for (var g = 1; g <= 4; g++) {
      var p = [];
      for (var i = 0; i < n; i++) {
        var a = -Math.PI / 2 + i * 2 * Math.PI / n;
        p.push((cx + Math.cos(a) * R * g / 4).toFixed(1) + ',' + (cy + Math.sin(a) * R * g / 4).toFixed(1));
      }
      grid += '<polygon points="' + p.join(' ') + '" fill="none" stroke="#26333d" stroke-width="1"/>';
    }
    for (i = 0; i < n; i++) {
      a = -Math.PI / 2 + i * 2 * Math.PI / n;
      var x = cx + Math.cos(a) * R, y = cy + Math.sin(a) * R;
      axes += '<line x1="' + cx + '" y1="' + cy + '" x2="' + x.toFixed(1) + '" y2="' + y.toFixed(1) + '" stroke="#26333d"/>';
      var v = Math.max(0, Math.min(100, r.radar[dims[i].key])) / 100;
      pts.push((cx + Math.cos(a) * R * v).toFixed(1) + ',' + (cy + Math.sin(a) * R * v).toFixed(1));
      var lx = cx + Math.cos(a) * (R + 20), ly = cy + Math.sin(a) * (R + 20);
      labels += '<text x="' + lx.toFixed(1) + '" y="' + (ly + 4).toFixed(1) + '" fill="#8fa1ad" font-size="11.5" text-anchor="middle">' + dims[i].label + '</text>';
    }
    $('radar').innerHTML = '<svg width="260" height="240">' + grid + axes +
      '<polygon points="' + pts.join(' ') + '" fill="rgba(53,211,154,.18)" stroke="#35d39a" stroke-width="2"/>' +
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
    $('scoreRating').innerHTML = r.rating.emoji + ' ' + esc(r.rating.label);
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
  /* 金额 → 星级。以当地同类学生水平为 3 星，别人只看得出高低，看不出具体数目。 */
  function stars(ratio) {
    var n = !isFinite(ratio) || ratio <= 0 ? 1
          : ratio < 0.60 ? 1 : ratio < 0.85 ? 2 : ratio < 1.15 ? 3 : ratio < 1.60 ? 4 : 5;
    return '★★★★★'.slice(0, n) + '☆☆☆☆☆'.slice(0, 5 - n);
  }

  function drawShare(hideMoney) {
    var r = refresh._r;
    // 先画在一张足够高的画布上，画完按实际内容裁掉多余部分 ——
    // 内容高度会随「有没有实习」变化，写死高度就会留一大块空白。
    var W = 780, H = 1400, dpr = 2;
    var cv = document.createElement('canvas');
    cv.width = W * dpr; cv.height = H * dpr;
    var x = cv.getContext('2d'); x.scale(dpr, dpr);
    var SANS = '"Microsoft YaHei","PingFang SC",-apple-system,sans-serif';
    var MONO = 'Consolas,"SF Mono",monospace';

    x.fillStyle = '#0f1418'; x.fillRect(0, 0, W, H);
    // 顶部渐变条
    var g = x.createLinearGradient(0, 0, W, 0);
    g.addColorStop(0, '#35d39a'); g.addColorStop(1, '#4aa3e0');
    x.fillStyle = g; x.fillRect(0, 0, W, 6);

    var P = 46, y = 62;
    x.fillStyle = '#dde5ea'; x.font = '700 30px ' + SANS;
    x.fillText('大学上得值不值', P, y); y += 30;
    x.fillStyle = '#61717c'; x.font = '400 14px ' + SANS;
    x.fillText('大学生活性价比测评 · 本科 / 授课型硕士', P, y); y += 40;

    // 学校 / 专业
    var school = (state.schoolName || '').trim() || '（未填写学校）';
    var major = (state.majorName || '').trim();
    x.fillStyle = '#dde5ea'; x.font = '600 22px ' + SANS;
    x.fillText(school.length > 22 ? school.slice(0, 22) + '…' : school, P, y); y += 26;
    x.fillStyle = '#8fa1ad'; x.font = '400 15px ' + SANS;
    var line2 = [major, r.country.label + ' · ' + r.region.label, r.loc.label]
      .filter(Boolean).join('  ·  ');
    x.fillText(line2.length > 40 ? line2.slice(0, 40) + '…' : line2, P, y); y += 34;

    // 主分数卡
    x.fillStyle = '#161d23'; roundRect(x, P, y, W - P * 2, 200, 14); x.fill();
    x.strokeStyle = '#26333d'; x.lineWidth = 1; x.stroke();
    x.textAlign = 'center';
    x.fillStyle = r.rating.color; x.font = '700 76px ' + MONO;
    x.fillText((Math.round(r.score * 100) / 100).toFixed(2), W / 2, y + 88);
    x.font = '600 24px ' + SANS;
    x.fillText(r.rating.emoji + ' ' + r.rating.label, W / 2, y + 128);
    x.fillStyle = '#61717c'; x.font = '400 13.5px ' + SANS;
    x.fillText('1.00 = 在当地花这个钱，过上了该有的大学生活', W / 2, y + 160);
    // 刻度条
    var bx = P + 40, bw = W - P * 2 - 80, by = y + 176;
    var bg = x.createLinearGradient(bx, 0, bx + bw, 0);
    ['#e0413a', '#e8763a', '#d9a13a', '#4aa3e0', '#2f9e6e', '#7b4fd9', '#c9971f']
      .forEach(function (c, i, a) { bg.addColorStop(i / (a.length - 1), c); });
    x.fillStyle = bg; roundRect(x, bx, by, bw, 6, 3); x.fill();
    var mx = bx + Math.max(0, Math.min(1, r.score / 3)) * bw;
    x.fillStyle = '#fff'; roundRect(x, mx - 2, by - 5, 4, 16, 2); x.fill();
    x.textAlign = 'left'; y += 232;

    // 五维
    x.fillStyle = '#8fa1ad'; x.font = '600 14px ' + SANS;
    x.fillText('五维评分（50 = 当地正常）', P, y); y += 16;
    D.RADAR_DIMS.forEach(function (d) {
      var v = Math.max(0, Math.min(100, r.radar[d.key]));
      x.fillStyle = '#61717c'; x.font = '400 13px ' + SANS;
      x.fillText(d.label, P, y + 14);
      x.fillStyle = '#1c252d'; roundRect(x, P + 46, y + 5, W - P * 2 - 100, 11, 5); x.fill();
      x.fillStyle = v >= 50 ? '#35d39a' : '#e0a33d';
      roundRect(x, P + 46, y + 5, (W - P * 2 - 100) * v / 100, 11, 5); x.fill();
      x.fillStyle = '#8fa1ad'; x.font = '600 12px ' + MONO;
      x.textAlign = 'right'; x.fillText(Math.round(v), W - P, y + 15); x.textAlign = 'left';
      y += 26;
    });
    y += 12;

    // 关键数字（可切换成星级，避免把具体收入晒出去）
    var cur = r.cur, facts;
    if (hideMoney) {
      var base = r.baselineMonthly || 1;
      facts = [
        ['月支出', stars(r.grossMonthly / base)],
        ['当地水平', '★★★☆☆'],
        r.intern.has ? ['实习收入', stars(r.intern.grossIncome / base)] : ['实习', '暂无'],
        ['净支出', stars(r.netMonthly / base)]
      ];
    } else {
      facts = [
        ['月支出', cur + M.fmt.money(r.grossMonthly)],
        ['当地基准', cur + M.fmt.money(r.baselineMonthly)]
      ];
      if (r.intern.has) facts.push(['实习月入', cur + M.fmt.money(r.intern.grossIncome)]);
      else facts.push(['实习', '暂无']);
      facts.push(['净支出', cur + M.fmt.money(r.netMonthly)]);
    }
    var fw = (W - P * 2 - 3 * 10) / 4;
    facts.slice(0, 4).forEach(function (ft, i) {
      var fx = P + i * (fw + 10);
      x.fillStyle = '#161d23'; roundRect(x, fx, y, fw, 62, 10); x.fill();
      x.strokeStyle = '#26333d'; x.stroke();
      x.textAlign = 'center';
      x.fillStyle = '#61717c'; x.font = '400 11px ' + SANS;
      x.fillText(ft[0], fx + fw / 2, y + 20);
      var t = String(ft[1]);
      var isStar = t.indexOf('★') >= 0 || t.indexOf('☆') >= 0;
      x.fillStyle = isStar ? '#e0a33d' : '#dde5ea';
      x.font = isStar ? '400 15px ' + SANS : (t.length > 10 ? '600 13px ' : '600 16px ') + MONO;
      x.fillText(t, fx + fw / 2, y + 44);
      x.textAlign = 'left';
    });
    y += 84;

    if (hideMoney) {
      x.fillStyle = '#61717c'; x.font = '400 11.5px ' + SANS;
      x.textAlign = 'center';
      x.fillText('★ 相对当地同类学生水平，三星 = 持平（已隐藏具体金额）', W / 2, y - 8);
      x.textAlign = 'left'; y += 8;
    }

    // 二维码 + 说明
    var qr = QR.make(SHARE_URL, { ecc: 'M' });
    var qs = 4, qFull = (qr.size + 8) * qs;
    x.fillStyle = '#161d23'; roundRect(x, P, y, W - P * 2, qFull + 28, 12); x.fill();
    x.strokeStyle = '#26333d'; x.stroke();
    QR.draw(x, qr, W - P - qFull - 14, y + 14, qs, '#0f1418', '#dde5ea');
    x.fillStyle = '#dde5ea'; x.font = '600 17px ' + SANS;
    x.fillText('扫码测测你的大学', P + 20, y + 40);
    x.fillStyle = '#8fa1ad'; x.font = '400 13px ' + SANS;
    x.fillText('住宿 · 地段 · 校园 · 前景 · 实习 · 花销', P + 20, y + 64);
    x.fillText('六维建模，境外按购买力折算', P + 20, y + 84);
    x.fillStyle = '#61717c'; x.font = '400 11.5px ' + SANS;
    x.fillText('全部本地计算，不上传任何数据', P + 20, y + 108);
    y += qFull + 42;

    /* 按实际内容裁剪 */
    var finalH = y + 40;
    var out = document.createElement('canvas');
    out.width = W * dpr; out.height = finalH * dpr;
    var o = out.getContext('2d');
    o.fillStyle = '#0f1418'; o.fillRect(0, 0, W * dpr, finalH * dpr);
    o.drawImage(cv, 0, 0, W * dpr, finalH * dpr, 0, 0, W * dpr, finalH * dpr);
    o.scale(dpr, dpr);
    o.fillStyle = '#3d4a54'; o.font = '400 11.5px ' + SANS;
    o.textAlign = 'center';
    o.fillText('本工具适用于本科生与授课型硕士 · 主观权重模型，仅供参考', W / 2, finalH - 16);

    return out;
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
  function renderShare() {
    var hide = $('hideMoney').checked;
    try { localStorage.setItem('uni-hide-money', hide ? '1' : '0'); } catch (e) {}
    var url;
    try { url = drawShare(hide).toDataURL('image/png'); }
    catch (e) { toast('生成失败：' + e.message); return null; }
    $('shareImg').src = url;
    return url;
  }

  try { $('hideMoney').checked = localStorage.getItem('uni-hide-money') === '1'; } catch (e) {}
  $('hideMoney').onchange = function () { var u = renderShare(); if (u) bindDownload(u); };

  function bindDownload(url) {
    $('btnDl').onclick = function () {
      var a = document.createElement('a');
      a.download = '大学性价比-' + ((state.schoolName || '测评').slice(0, 12)) + '.png';
      a.href = url; a.click();
    };
  }

  $('btnShare').onclick = function () {
    var url = renderShare();
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
