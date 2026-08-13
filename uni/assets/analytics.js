/* ===========================================================================
 * analytics.js —— 访问统计
 * ---------------------------------------------------------------------------
 * 只上报「哪个页面被打开了」，绝不上报用户填写的任何内容。
 *
 * 这一点必须守死：这个工具会收集学校、专业、花销，甚至「长期情绪很差 /
 * 已经在就医」这种选项。所有答案只存在浏览器本地 localStorage，
 * 任何情况下都不进入统计请求。下面的代码里没有、也不允许出现
 * 读取 state / 表单值的逻辑。
 *
 * 配置：把 PROVIDER 和对应的 ID 填上即可，9 个主题页共用这一处。
 * =========================================================================== */
(function () {
  'use strict';

  var CFG = {
    // 'cf'（Cloudflare Web Analytics）| 'ga'（Google Analytics 4）
    // | 'umami' | 'baidu' | 'none'
    provider: 'cf',

    // 与 yuhangwu.com 主站同一个 token，统计合并在一处看
    cfToken:  '0a7f28af98dc4cf585b39b8d3fb0d54f',
    gaId:     '',                       // 形如 G-XXXXXXXXXX
    umamiSrc: '', umamiId: '',          // 自建 Umami
    baiduId:  ''                        // 百度统计的 hm.js? 后面那串
  };

  // 允许页面在引入前用 window.UNI_ANALYTICS 覆盖配置
  if (window.UNI_ANALYTICS) {
    Object.keys(window.UNI_ANALYTICS).forEach(function (k) { CFG[k] = window.UNI_ANALYTICS[k]; });
  }

  function inject(src, attrs) {
    var s = document.createElement('script');
    s.async = true; s.defer = true; s.src = src;
    if (attrs) Object.keys(attrs).forEach(function (k) { s.setAttribute(k, attrs[k]); });
    document.head.appendChild(s);
    return s;
  }

  switch (CFG.provider) {
    case 'cf':
      if (!CFG.cfToken) break;
      inject('https://static.cloudflareinsights.com/beacon.min.js',
             { 'data-cf-beacon': JSON.stringify({ token: CFG.cfToken }) });
      break;

    case 'ga':
      if (!CFG.gaId) break;
      inject('https://www.googletagmanager.com/gtag/js?id=' + encodeURIComponent(CFG.gaId));
      window.dataLayer = window.dataLayer || [];
      window.gtag = function () { window.dataLayer.push(arguments); };
      window.gtag('js', new Date());
      // anonymize_ip：不需要精确 IP，只要知道有人来过
      window.gtag('config', CFG.gaId, { anonymize_ip: true });
      break;

    case 'umami':
      if (!CFG.umamiSrc || !CFG.umamiId) break;
      inject(CFG.umamiSrc, { 'data-website-id': CFG.umamiId });
      break;

    case 'baidu':
      if (!CFG.baiduId) break;
      window._hmt = window._hmt || [];
      inject('https://hm.baidu.com/hm.js?' + encodeURIComponent(CFG.baiduId));
      break;

    default:
      break;   // 'none'：什么都不做，页面保持零外部请求
  }

  // 供页面显示当前是否开启了统计，用来保证界面上的隐私说明始终属实
  window.UNI_ANALYTICS_ON = CFG.provider !== 'none';
})();
