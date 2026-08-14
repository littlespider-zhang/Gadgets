module.exports = async ({ app, quickAddApi }) => {
        // 加粗希腊字母
const boldLetters = [
    ["								𝛂	~	alpha", "\\alpha"],
    ["								𝛃	~	beta", "\\beta"],
    ["								𝛄	~	gamma", "\\gamma"],
    ["								𝛅	~	delta", "\\delta"],
    ["								𝛆	~	epsilon", "\\epsilon"],
    ["								𝛇	~	zeta", "\\zeta"],
    ["								𝛈	~	eta", "\\eta"],
    ["								𝛉	~	theta", "\\theta"],
    ["								𝛊	~	iota", "\\iota"],
    ["								𝛋	~	kappa", "\\kappa"],
    ["								𝛌	~	lambda", "\\lambda"],
    ["								𝛍	~	mu", "\\mu"],
    ["								𝛎	~	nu", "\\nu"],
    ["								𝛏	~	xi", "\\xi"],
    ["								𝛐	~	omicron", "o"],
    ["								𝛑	~	pi", "\\pi"],
    ["								𝛒	~	rho", "\\rho"],
    ["								𝛔	~	sigma", "\\sigma"],
    ["								𝛕	~	tau", "\\tau"],
    ["								𝛖	~	upsilon", "\\upsilon"],
    ["								𝛗	~	phi", "\\phi"],
    ["								𝛘	~	chi", "\\chi"],
    ["								𝛙	~	psi", "\\psi"],
    ["								𝛚	~	omega", "\\omega"],
];
	
	// 普通希腊字母
    const letters = [
	["								α	~	alpha", "\\alpha"],
    ["								β	~	beta", "\\beta"],
    ["								γ	~	gamma", "\\gamma"],
    ["								δ	~	delta", "\\delta"],
    ["								ε	~	epsilon", "\\epsilon"],
    ["								ζ	~	zeta", "\\zeta"],
    ["								η	~	eta", "\\eta"],
    ["								θ	~	theta", "\\theta"],
    ["								ι	~	iota", "\\iota"],
    ["								κ	~	kappa", "\\kappa"],
    ["								λ	~	lambda", "\\lambda"],
    ["								μ	~	mu", "\\mu"],
    ["								ν	~	nu", "\\nu"],
    ["								ξ	~	xi", "\\xi"],
    ["								ο	~	omicron", "o"],
    ["								π	~	pi", "\\pi"],
    ["								ρ	~	rho", "\\rho"],
    ["								σ	~	sigma", "\\sigma"],
    ["								τ	~	tau", "\\tau"],
    ["								υ	~	upsilon", "\\upsilon"],
    ["								φ	~	phi", "\\phi"],
    ["								χ	~	chi", "\\chi"],
    ["								ψ	~	psi", "\\psi"],
    ["								ω	~	omega", "\\omega"],
];



    // QuickAdd 对话框中显示的内容
    const displayItems = [
		...boldLetters.map(([symbol]) => `${symbol}`),
        ...letters.map(([symbol]) => `${symbol}`),
    ];

    // 实际插入 Obsidian 的 LaTeX
    const actualItems = [
		...boldLetters.map(([, latex]) => `$\\boldsymbol{${latex}}$`),
        ...letters.map(([, latex]) => `$${latex}$`),
    ];

    // 弹出选择框
    const selected = await quickAddApi.suggester(
        displayItems,
        actualItems,
        "Choose letters"
    );

    if (selected == null) return;

    // 获取当前编辑器
    const editor = app.workspace.activeEditor?.editor;

    if (!editor) {
        await quickAddApi.infoDialog(
            "无法插入",
            "请先打开一个 Markdown 笔记，并把光标放在需要插入的位置。"
        );
        return;
    }

    // 插入到当前光标位置
    const cursor = editor.getCursor();
    editor.replaceRange(selected, cursor);
};