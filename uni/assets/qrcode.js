/* ===========================================================================
 * qrcode.js —— 零依赖 QR Code (Model 2) 编码器
 * ---------------------------------------------------------------------------
 * 为什么要自己写：整个工具必须纯前端、可离线、不发任何网络请求，
 * 所以不能用 CDN 上的二维码库，也不能调用在线生成接口。
 *
 * 实现范围：
 *   · 字节模式（8-bit Byte），输入按 UTF-8 编码，足够放 URL
 *   · 纠错等级 L / M / Q / H
 *   · 版本 1 ~ 15（版本 15-M 可放约 412 字节，远超本工具需要）
 *   · Reed-Solomon 纠错（GF(256)，本原多项式 0x11D）
 *   · 8 种掩码全部生成并按标准罚分规则择优
 *
 * 用法：
 *   var qr = UNI_QR.make('https://example.com/', { ecc: 'M' });
 *   qr.size            → 模块边长
 *   qr.isDark(r, c)    → 该模块是否为黑
 *   UNI_QR.draw(ctx, qr, x, y, scale, dark, light)   // 画到 canvas
 * =========================================================================== */

(function (root) {
  'use strict';

  /* ---------------- GF(256) ---------------- */
  var EXP = new Array(256), LOG = new Array(256);
  (function () {
    for (var i = 0; i < 8; i++) EXP[i] = 1 << i;
    // 本原多项式 x^8 + x^4 + x^3 + x^2 + 1 (0x11D) 对应的递推
    for (i = 8; i < 256; i++) EXP[i] = EXP[i - 4] ^ EXP[i - 5] ^ EXP[i - 6] ^ EXP[i - 8];
    for (i = 0; i < 255; i++) LOG[EXP[i]] = i;
  })();

  function gmul(a, b) {
    if (a === 0 || b === 0) return 0;
    return EXP[(LOG[a] + LOG[b]) % 255];
  }

  /* 生成多项式 g(x) = ∏ (x - α^i) */
  function rsPoly(n) {
    var p = [1];
    for (var i = 0; i < n; i++) {
      var q = new Array(p.length + 1);
      for (var j = 0; j < q.length; j++) q[j] = 0;
      for (j = 0; j < p.length; j++) {
        q[j]     ^= gmul(p[j], 1);
        q[j + 1] ^= gmul(p[j], EXP[i]);
      }
      p = q;
    }
    return p;
  }

  /* 对 data 做多项式除法，返回 n 个纠错码字 */
  function rsEncode(data, n) {
    var gen = rsPoly(n);
    var res = data.slice().concat(new Array(n));
    for (var i = data.length; i < res.length; i++) res[i] = 0;
    for (i = 0; i < data.length; i++) {
      var coef = res[i];
      if (coef === 0) continue;
      for (var j = 1; j < gen.length; j++) res[i + j] ^= gmul(gen[j], coef);
    }
    return res.slice(data.length);
  }

  /* ---------------- 版本表 ----------------
   * 每个版本 4 组（L/M/Q/H），每组为 [块数, 总码字, 数据码字, ...] 的扁平数组
   */
  var RS_BLOCKS = [
    /* v1  */[[1,26,19],[1,26,16],[1,26,13],[1,26,9]],
    /* v2  */[[1,44,34],[1,44,28],[1,44,22],[1,44,16]],
    /* v3  */[[1,70,55],[1,70,44],[2,35,17],[2,35,13]],
    /* v4  */[[1,100,80],[2,50,32],[2,50,24],[4,25,9]],
    /* v5  */[[1,134,108],[2,67,43],[2,33,15,2,34,16],[2,33,11,2,34,12]],
    /* v6  */[[2,86,68],[4,43,27],[4,43,19],[4,43,15]],
    /* v7  */[[2,98,78],[4,49,31],[2,32,14,4,33,15],[4,39,13,1,40,14]],
    /* v8  */[[2,121,97],[2,60,38,2,61,39],[4,40,18,2,41,19],[4,40,14,2,41,15]],
    /* v9  */[[2,146,116],[3,58,36,2,59,37],[4,36,16,4,37,17],[4,36,12,4,37,13]],
    /* v10 */[[2,86,68,2,87,69],[4,69,43,1,70,44],[6,43,19,2,44,20],[6,43,15,2,44,16]],
    /* v11 */[[4,101,81],[1,80,50,4,81,51],[4,50,22,4,51,23],[3,36,12,8,37,13]],
    /* v12 */[[2,116,92,2,117,93],[6,58,36,2,59,37],[4,46,20,6,47,21],[7,42,14,4,43,15]],
    /* v13 */[[4,133,107],[8,59,37,1,60,38],[8,44,20,4,45,21],[12,33,11,4,34,12]],
    /* v14 */[[3,145,115,1,146,116],[4,64,40,5,65,41],[11,36,16,5,37,17],[11,36,12,5,37,13]],
    /* v15 */[[5,109,87,1,110,88],[5,65,41,5,66,42],[5,54,24,7,55,25],[11,36,12,7,37,13]]
  ];

  var ALIGN = [
    [], [6,18], [6,22], [6,26], [6,30], [6,34], [6,22,38], [6,24,42],
    [6,26,46], [6,28,50], [6,30,54], [6,32,58], [6,34,62], [6,26,46,66], [6,26,48,70]
  ];

  var ECC_INDEX = { L: 0, M: 1, Q: 2, H: 3 };
  var ECC_BITS  = { L: 1, M: 0, Q: 3, H: 2 };   // 格式信息里的编码，注意不是 0123

  function blocksOf(version, ecc) {
    var raw = RS_BLOCKS[version - 1][ECC_INDEX[ecc]];
    var out = [];
    for (var i = 0; i < raw.length; i += 3) {
      for (var k = 0; k < raw[i]; k++) out.push({ total: raw[i + 1], data: raw[i + 2] });
    }
    return out;
  }
  function dataCapacity(version, ecc) {
    return blocksOf(version, ecc).reduce(function (s, b) { return s + b.data; }, 0);
  }

  /* ---------------- BCH（格式信息 / 版本信息） ---------------- */
  function bchDigit(d) { var n = 0; while (d !== 0) { n++; d >>>= 1; } return n; }
  function bchFormat(data) {
    var G15 = 0x537, MASK = 0x5412, d = data << 10;
    while (bchDigit(d) - bchDigit(G15) >= 0) d ^= (G15 << (bchDigit(d) - bchDigit(G15)));
    return ((data << 10) | d) ^ MASK;
  }
  function bchVersion(data) {
    var G18 = 0x1F25, d = data << 12;
    while (bchDigit(d) - bchDigit(G18) >= 0) d ^= (G18 << (bchDigit(d) - bchDigit(G18)));
    return (data << 12) | d;
  }

  /* ---------------- UTF-8 ---------------- */
  function utf8Bytes(str) {
    var out = [];
    for (var i = 0; i < str.length; i++) {
      var c = str.charCodeAt(i);
      if (c < 0x80) out.push(c);
      else if (c < 0x800) out.push(0xC0 | (c >> 6), 0x80 | (c & 63));
      else if (c >= 0xD800 && c <= 0xDBFF && i + 1 < str.length) {
        var c2 = str.charCodeAt(++i);
        var cp = 0x10000 + ((c - 0xD800) << 10) + (c2 - 0xDC00);
        out.push(0xF0 | (cp >> 18), 0x80 | ((cp >> 12) & 63), 0x80 | ((cp >> 6) & 63), 0x80 | (cp & 63));
      } else out.push(0xE0 | (c >> 12), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63));
    }
    return out;
  }

  /* ---------------- 位缓冲 ---------------- */
  function BitBuffer() { this.buf = []; this.len = 0; }
  BitBuffer.prototype.put = function (num, bits) {
    for (var i = 0; i < bits; i++) this.putBit(((num >>> (bits - i - 1)) & 1) === 1);
  };
  BitBuffer.prototype.putBit = function (b) {
    var idx = Math.floor(this.len / 8);
    if (this.buf.length <= idx) this.buf.push(0);
    if (b) this.buf[idx] |= (0x80 >>> (this.len % 8));
    this.len++;
  };

  /* ---------------- 掩码 ---------------- */
  var MASKS = [
    function (r, c) { return (r + c) % 2 === 0; },
    function (r)    { return r % 2 === 0; },
    function (r, c) { return c % 3 === 0; },
    function (r, c) { return (r + c) % 3 === 0; },
    function (r, c) { return (Math.floor(r / 2) + Math.floor(c / 3)) % 2 === 0; },
    function (r, c) { return (r * c) % 2 + (r * c) % 3 === 0; },
    function (r, c) { return ((r * c) % 2 + (r * c) % 3) % 2 === 0; },
    function (r, c) { return ((r + c) % 2 + (r * c) % 3) % 2 === 0; }
  ];

  /* 标准四条罚分规则 */
  function penalty(m, n) {
    var p = 0, r, c, i, run, dark = 0;

    // 规则1：同色连续 5 个以上
    for (r = 0; r < n; r++) {
      run = 1;
      for (c = 1; c < n; c++) {
        if (m[r][c] === m[r][c - 1]) { run++; if (c === n - 1 && run >= 5) p += 3 + (run - 5); }
        else { if (run >= 5) p += 3 + (run - 5); run = 1; }
      }
    }
    for (c = 0; c < n; c++) {
      run = 1;
      for (r = 1; r < n; r++) {
        if (m[r][c] === m[r - 1][c]) { run++; if (r === n - 1 && run >= 5) p += 3 + (run - 5); }
        else { if (run >= 5) p += 3 + (run - 5); run = 1; }
      }
    }

    // 规则2：2×2 同色块
    for (r = 0; r < n - 1; r++)
      for (c = 0; c < n - 1; c++)
        if (m[r][c] === m[r][c + 1] && m[r][c] === m[r + 1][c] && m[r][c] === m[r + 1][c + 1]) p += 3;

    // 规则3：形如 1:1:3:1:1 且一侧有 4 个浅色的图案
    var pat = [true, false, true, true, true, false, true];
    function match(get) {
      var cnt = 0, k, j, ok;
      for (k = 0; k <= n - 7; k++) {
        ok = true;
        for (j = 0; j < 7; j++) if (get(k + j) !== pat[j]) { ok = false; break; }
        if (!ok) continue;
        var before = true, after = true;
        for (j = 1; j <= 4; j++) { if (k - j < 0) break; if (get(k - j)) { before = false; break; } }
        for (j = 7; j < 11; j++) { if (k + j >= n) break; if (get(k + j)) { after = false; break; } }
        if (before || after) cnt++;
      }
      return cnt;
    }
    for (r = 0; r < n; r++) p += 40 * match(function (x) { return m[r][x]; });
    for (c = 0; c < n; c++) p += 40 * match(function (x) { return m[x][c]; });

    // 规则4：深色比例偏离 50%
    for (r = 0; r < n; r++) for (c = 0; c < n; c++) if (m[r][c]) dark++;
    var ratio = dark * 100 / (n * n);
    p += Math.floor(Math.abs(ratio - 50) / 5) * 10;

    return p;
  }

  /* ---------------- 主流程 ---------------- */
  function make(text, opts) {
    opts = opts || {};
    var ecc = opts.ecc || 'M';
    if (!(ecc in ECC_INDEX)) ecc = 'M';

    var bytes = utf8Bytes(String(text));

    // 选最小够用的版本
    var version = 0;
    for (var v = Math.max(1, opts.minVersion || 1); v <= RS_BLOCKS.length; v++) {
      var lenBits = v < 10 ? 8 : 16;
      // 4 位模式指示 + 字符数 + 数据，向上取整到字节
      if (dataCapacity(v, ecc) * 8 >= 4 + lenBits + bytes.length * 8) { version = v; break; }
    }
    if (!version) throw new Error('内容过长，超出版本 ' + RS_BLOCKS.length + ' 在纠错等级 ' + ecc + ' 下的容量');

    var size = version * 4 + 17;
    var lenBits = version < 10 ? 8 : 16;

    /* 1. 组数据位流 */
    var bb = new BitBuffer();
    bb.put(4, 4);                       // 字节模式
    bb.put(bytes.length, lenBits);
    for (var i = 0; i < bytes.length; i++) bb.put(bytes[i], 8);

    var totalData = dataCapacity(version, ecc);
    // 终止符最多 4 位
    for (i = 0; i < 4 && bb.len < totalData * 8; i++) bb.putBit(false);
    while (bb.len % 8 !== 0) bb.putBit(false);
    var pad = [0xEC, 0x11], pi = 0;
    while (bb.buf.length < totalData) bb.buf.push(pad[pi++ % 2]);

    /* 2. 分块 + 纠错 */
    var blocks = blocksOf(version, ecc);
    var dataBlocks = [], eccBlocks = [], off = 0;
    blocks.forEach(function (b) {
      var d = bb.buf.slice(off, off + b.data); off += b.data;
      dataBlocks.push(d);
      eccBlocks.push(rsEncode(d, b.total - b.data));
    });

    /* 3. 交错 */
    var out = [], maxD = 0, maxE = 0;
    dataBlocks.forEach(function (d) { maxD = Math.max(maxD, d.length); });
    eccBlocks.forEach(function (e) { maxE = Math.max(maxE, e.length); });
    for (i = 0; i < maxD; i++) for (var b = 0; b < dataBlocks.length; b++)
      if (i < dataBlocks[b].length) out.push(dataBlocks[b][i]);
    for (i = 0; i < maxE; i++) for (b = 0; b < eccBlocks.length; b++)
      if (i < eccBlocks[b].length) out.push(eccBlocks[b][i]);

    /* 4. 铺模块 */
    function blank() {
      var m = [];
      for (var r = 0; r < size; r++) { m.push([]); for (var c = 0; c < size; c++) m[r].push(null); }
      return m;
    }
    var reserved = blank();

    function setupFunctions(m, res) {
      // 定位图案 + 分隔符
      [[0, 0], [size - 7, 0], [0, size - 7]].forEach(function (p) {
        for (var r = -1; r <= 7; r++) for (var c = -1; c <= 7; c++) {
          var rr = p[0] + r, cc = p[1] + c;
          if (rr < 0 || rr >= size || cc < 0 || cc >= size) continue;
          var on = (r >= 0 && r <= 6 && (c === 0 || c === 6)) ||
                   (c >= 0 && c <= 6 && (r === 0 || r === 6)) ||
                   (r >= 2 && r <= 4 && c >= 2 && c <= 4);
          m[rr][cc] = on; res[rr][cc] = true;
        }
      });
      /* 校正图案必须画在定时图案**之前**。
       * 跳过条件靠「中心格已被占用」判断是否与定位图案重叠，如果先画了
       * 定时图案，第 6 行/列上的校正图案中心（v7 起就有，如 v8 的 (6,24)）
       * 会被误判成重叠而整个漏掉，数据位填进本该是校正图案的位置 ——
       * v1~v6 看不出问题，v7 起二维码直接扫不出来。 */
      var pos = ALIGN[version - 1];
      for (var a = 0; a < pos.length; a++) for (var b2 = 0; b2 < pos.length; b2++) {
        var row = pos[a], col = pos[b2];
        if (m[row][col] !== null) continue;   // 与定位图案重叠则跳过
        for (var r2 = -2; r2 <= 2; r2++) for (var c2 = -2; c2 <= 2; c2++) {
          m[row + r2][col + c2] =
            Math.max(Math.abs(r2), Math.abs(c2)) !== 1;
          res[row + r2][col + c2] = true;
        }
      }
      // 定时图案
      for (var i = 8; i < size - 8; i++) {
        if (m[6][i] === null) { m[6][i] = i % 2 === 0; res[6][i] = true; }
        if (m[i][6] === null) { m[i][6] = i % 2 === 0; res[i][6] = true; }
      }
      // 固定黑模块
      m[size - 8][8] = true; res[size - 8][8] = true;

      /* 预留格式信息区。
       * 关键：除了在 res 里标记，还必须把 m 置成非 null 占位 ——
       * placeData 靠 m[r][c] === null 判断空位，只标 res 的话数据位
       * 会被写进格式信息区，整个位流后移，解出来全是乱码。 */
      function reserve(r, c) { res[r][c] = true; if (m[r][c] === null) m[r][c] = false; }
      for (i = 0; i <= 8; i++) { if (i !== 6) { reserve(8, i); reserve(i, 8); } }
      for (i = 0; i < 8; i++) { reserve(8, size - 1 - i); reserve(size - 1 - i, 8); }
      reserve(8, 6); reserve(6, 8);

      // 版本信息区（v7+）
      if (version >= 7) {
        for (i = 0; i < 6; i++) for (var j = 0; j < 3; j++) {
          reserve(size - 11 + j, i); reserve(i, size - 11 + j);
        }
      }
    }

    var baseM = blank();
    setupFunctions(baseM, reserved);

    // 数据按之字形填入（跳过第 6 列）
    function placeData(m) {
      var bitIdx = 0, dir = -1, row = size - 1;
      for (var col = size - 1; col > 0; col -= 2) {
        if (col === 6) col--;
        while (true) {
          for (var k = 0; k < 2; k++) {
            var c = col - k;
            if (m[row][c] === null) {
              var dark = false;
              if (bitIdx < out.length * 8)
                dark = ((out[bitIdx >>> 3] >>> (7 - (bitIdx & 7))) & 1) === 1;
              m[row][c] = dark;
              bitIdx++;
            }
          }
          row += dir;
          if (row < 0 || row >= size) { row -= dir; dir = -dir; break; }
        }
      }
    }

    function applyMask(m, maskIdx) {
      var o = [];
      for (var r = 0; r < size; r++) {
        o.push([]);
        for (var c = 0; c < size; c++) {
          var v2 = m[r][c];
          if (!reserved[r][c] && MASKS[maskIdx](r, c)) v2 = !v2;
          o[r].push(!!v2);
        }
      }
      return o;
    }

    function putFormat(m, maskIdx) {
      var bits = bchFormat((ECC_BITS[ecc] << 3) | maskIdx);
      var i, on;
      // 竖向：左上角自上而下 + 左下角
      for (i = 0; i < 15; i++) {
        on = ((bits >> i) & 1) === 1;
        if (i < 6)      m[i][8] = on;
        else if (i < 8) m[i + 1][8] = on;         // 跳过第 6 行（定时图案）
        else            m[size - 15 + i][8] = on;
      }
      // 横向：右上角自右向左 + 左上角
      // i === 8 这一档要额外 +1 跳过第 6 列，漏掉的话有一位会落在定时图案上
      for (i = 0; i < 15; i++) {
        on = ((bits >> i) & 1) === 1;
        if (i < 8)      m[8][size - i - 1] = on;
        else if (i < 9) m[8][15 - i - 1 + 1] = on;
        else            m[8][15 - i - 1] = on;
      }
      m[size - 8][8] = true;
    }

    function putVersion(m) {
      if (version < 7) return;
      var bits = bchVersion(version);
      for (var i = 0; i < 18; i++) {
        var on = ((bits >> i) & 1) === 1;
        m[Math.floor(i / 3)][size - 11 + (i % 3)] = on;
        m[size - 11 + (i % 3)][Math.floor(i / 3)] = on;
      }
    }

    // 先填数据，再对 8 种掩码各评一次分
    var filled = [];
    for (var r0 = 0; r0 < size; r0++) filled.push(baseM[r0].slice());
    placeData(filled);

    var best = null, bestP = Infinity, bestMask = 0;
    for (var mk = 0; mk < 8; mk++) {
      var cand = applyMask(filled, mk);
      putFormat(cand, mk);
      putVersion(cand);
      var p = penalty(cand, size);
      if (p < bestP) { bestP = p; best = cand; bestMask = mk; }
    }

    return {
      size: size, version: version, ecc: ecc, mask: bestMask, penalty: bestP,
      modules: best,
      isDark: function (r, c) { return !!best[r][c]; }
    };
  }

  /* ---------------- 画到 canvas ---------------- */
  function draw(ctx, qr, x, y, scale, dark, light) {
    var q = 4;                       // 静区，标准要求 4 个模块
    var full = (qr.size + q * 2) * scale;
    if (light !== 'none') {
      ctx.fillStyle = light || '#fff';
      ctx.fillRect(x, y, full, full);
    }
    ctx.fillStyle = dark || '#000';
    for (var r = 0; r < qr.size; r++)
      for (var c = 0; c < qr.size; c++)
        if (qr.isDark(r, c))
          ctx.fillRect(x + (c + q) * scale, y + (r + q) * scale, scale, scale);
    return full;
  }

  root.UNI_QR = { make: make, draw: draw, capacity: dataCapacity };
  if (typeof module !== 'undefined' && module.exports) module.exports = root.UNI_QR;

})(typeof window !== 'undefined' ? window : globalThis);
