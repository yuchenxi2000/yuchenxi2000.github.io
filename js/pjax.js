(function() {
    // eslint-disable-next-line no-unused-vars
    let pjax;

    function initPjax() {
        try {
            const Pjax = window.Pjax || function() {};
            pjax = new Pjax({
                selectors: [
                    '[data-pjax]',
                    '.pjax-reload',
                    'head title',
                    '.columns',
                    '.navbar-start',
                    '.navbar-end',
                    '.searchbox link',
                    '.searchbox script',
                    '#back-to-top',
                    '#comments link',
                    '#comments script'
                ],
                cacheBust: false
            });
        } catch (e) {
            console.warn('PJAX error: ' + e);
        }
    }

    // // Listen for start of Pjax
    // document.addEventListener('pjax:send', function() {
    //     return;
    //     // TODO pace start loading animation
    // })

    // Listen for completion of Pjax
    document.addEventListener('pjax:complete', () => {
        // Plugin [MathJax] reload logic
        if (window.MathJax) {
            try {
                var mj = window.MathJax;
                if (mj.typesetClear) mj.typesetClear();
                // MathJax v3.2.x 的 bug: typesetClear()/document.clear() 不会清空
                // inputJax._parseOptions.tags 里的 labels/allLabels，导致 PJAX 返回时
                // \tag{1} 等自动编号被重复注册。需手动清除所有 inputJax 缓存中的 tags。
                var clearTags = function(inputJax) {
                    var tags = inputJax && inputJax._parseOptions && inputJax._parseOptions.tags;
                    if (tags) { tags.labels = {}; tags.allLabels = {}; }
                };
                var inputJaxList = mj.startup && mj.startup.input;
                if (inputJaxList) inputJaxList.forEach(clearTags);
                var doc = mj.startup && mj.startup.document;
                if (doc && doc.inputJax) doc.inputJax.forEach(clearTags);
                // CHTML 输出处理器内部缓存了一份 inputJax
                try {
                    var outDoc = doc && doc.outputJax && doc.outputJax.document;
                    if (outDoc && outDoc.inputJax) outDoc.inputJax.forEach(clearTags);
                } catch(e) {}
                if (mj.typesetPromise) mj.typesetPromise();
            } catch (e) {
                console.error('MathJax reload error:', e);
            }
        }
        // Plugin [Busuanzi] reload logic
        if (window.bszCaller && window.bszTag) {
            window.bszCaller.fetch('//busuanzi.ibruce.info/busuanzi?jsonpCallback=BusuanziCallback', a => {
                window.bszTag.texts(a);
                window.bszTag.shows();
            });
        }

        // TODO pace stop loading animation
    });

    document.addEventListener('DOMContentLoaded', () => initPjax());
}());
