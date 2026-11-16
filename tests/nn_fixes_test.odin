package tests

import l "../wotan/linalg"
import nn "../wotan/nn"
import tok "../wotan/nn/tokenizers"
import t "../wotan/tensor"
import "base:runtime"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:os"
import "core:slice"

// Regression tests for wotan/nn issues found while writing the nn guide:
// constant op inputs made by forward passes (position ids, zero RNN / Mamba
// states, the [CLS] gather ids, the positional encoding, the MoE top-k mask)
// were allocated on the heap even inside a graph scope, so graph_end left one
// tensor per constant per step behind; MoE without aux_loss leaked its router
// softmax, and as the first layer of a model (input without gradient) its
// token matrix; the probabilistic losses leaked the intermediates computed
// from data inputs (a fixed variance, scale or ν); a checkpoint whose layer
// header had an invalid configuration (e.g. 0 attention heads) stopped
// reading and could load as ok. Found with the tokenizers' part of the guide:
// the WordPiece cleaner kept private-use and unassigned characters (as [UNK])
// that Hugging Face's BertTokenizer removes with the other C* categories.
// Later: bpe_tokenize put [CLS] / [SEP] ids (0 = "!") into GPT-2 input; GPT
// with use_flash ignored an explicit attention mask; adam / sgd_zero_grad
// zeroed the gradients instead of dropping them, so parameters without a
// gradient still moved (momentum, weight decay); checkpoints lost the frozen
// flags (and with them the Adam moments); the Mamba layer had no input
// normalisation (outputs exploded on large inputs); the GRU had no hidden-path
// n-gate bias (PyTorch's bias_hh n part, inside the reset gate); GPT had
// exact GELU only (GPT-2 / DistilGPT-2 use the tanh approximation);
// reparameterize required grad even for inputs without it.
// Every check runs under a Tracking_Allocator on the default heap allocator.

@(private = "file")
nf_ids :: proc(vals: []f64, batch, seq: int) -> ^t.Tensor {
	m := l.matrix_new(f64, 1, batch * seq)
	copy(m.data, vals)
	x := t.tensor_new(m, false)
	x.shape = {batch, seq, 1, 1}
	return x
}

@(private = "file")
nf_seq :: proc(batch, seq, d: int) -> ^t.Tensor {
	m := l.matrix_new(f64, 1, batch * seq * d)
	for &v, i in m.data {v = 0.1 * f64(i % 7) - 0.3}
	x := t.tensor_new(m, false)
	x.shape = {batch, seq, d, 1}
	return x
}

@(private = "file")
NF_TARGETS :: []int{4, 2, 7, 0, 0, 1, 5, 9, 2, 1, 6, 3}

@(private = "file")
nf_gpt: nn.GPTModel
@(private = "file")
nf_bert: nn.BERTModel
@(private = "file")
nf_model: ^nn.Sequential
@(private = "file")
nf_ids_t, nf_seg_t, nf_x: ^t.Tensor

// Live heap allocations after each of 4 training steps (scoped: inside a
// graph scope, else freed with tensor_free_graph); all equal = no leak
@(private = "file")
nf_steps :: proc(
	track: ^mem.Tracking_Allocator,
	scoped: bool,
	loss_of: proc() -> ^t.Tensor,
) -> [4]int {
	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)
	counts: [4]int
	for i in 0 ..< 4 {
		if scoped {t.graph_begin(&scope)}
		loss := loss_of()
		t.tensor_backward(loss)
		if scoped {t.graph_end(&scope)} else {t.tensor_free_graph(loss)}
		counts[i] = len(track.allocation_map)
	}
	return counts
}

// Reference values (torch 2.1.0, see the Mamba and optimizer checks)
@(private = "file")
NF_MAMBA_OUT_1 :: [?]f64 {
	0.058479590481694654,
	1.2131926293919633,
	1.3306647354119105,
	0.5033410494374555,
	-1.0223847683855922,
	-1.1538665614081305,
	-0.48098887233075094,
	0.42589315682269346,
	0.53793765468160348,
	0.58050135238531708,
	0.30260814275638426,
	-0.048218360214594136,
	-0.77803426276933352,
	-0.46192351625995376,
	0.27410377527059049,
	1.0638606147510858,
	1.0551541760204872,
	0.43862124006560066,
	-0.60169624695115087,
	-1.1646832725515495,
	-1.1148183723436376,
	0.28447010950717899,
	1.4035980746007528,
	1.4423540305006179,
	-0.080423741227698131,
	-0.91150082614093264,
	-0.9734723794103558,
	-0.2732489166885399,
	0.12650434450836873,
	0.59409301733441777,
	0.58253914382156502,
	0.26233943421471789,
	-0.65302071562931396,
	-0.68095198483472252,
	-0.29093254482759906,
	0.49004556600114768,
	0.83999557684671822,
	1.0168035196417196,
	0.37084984696721146,
	-0.65385301353946157,
}
@(private = "file")
NF_MAMBA_DGAMMA_1 :: [?]f64 {
	0.10176530788438443,
	0.063847510902573956,
	-0.0055170008687449121,
	0.089746736651219211,
}
@(private = "file")
NF_MAMBA_DTHETA_A_1 :: [?]f64 {
	-0.0010988518826532943,
	-0.0035420160855249051,
	-0.0044965139966910992,
	0.0011782591184675069,
	0.0015141376614623758,
	0.00047044333589200537,
	-0.0051749730856642728,
	-0.0016009843119260873,
	0.0030872432432427774,
	-0.0040056093046908796,
	-0.0037552050251448833,
	-0.00071373593820793231,
}
@(private = "file")
NF_MAMBA_OUT_10 :: [?]f64 {
	2.7405158526930591,
	13.065372114516812,
	12.891481534611831,
	3.3258237808022328,
	-7.5268071138625645,
	-10.471468486481983,
	-5.5169751628747363,
	1.873262393996141,
	5.6441647762005935,
	5.02973051328373,
	1.315658095503959,
	-2.8962494816605338,
	-5.7425732274782701,
	-4.0058918643025736,
	1.8469418109837652,
	8.3580826570880333,
	9.460603219879717,
	2.6480154130874292,
	-8.1671610963067902,
	-13.91801369443848,
	-9.1716544809743006,
	3.6300746191445343,
	13.523901929561463,
	12.683316613576952,
	2.1495600153209664,
	-7.7855710199387556,
	-10.209579832721365,
	-4.9473689484150469,
	1.8861870918234451,
	5.772885989519712,
	4.8909034157767053,
	1.0489410510493764,
	-3.6851016198326119,
	-5.6720687716046152,
	-3.6334774984641851,
	2.4127110657346127,
	8.3654455255735805,
	9.2656855976025785,
	2.0604383000227102,
	-8.7092436633165118,
}
@(private = "file")
NF_MAMBA_DGAMMA_10 :: [?]f64 {
	0.8033363737897502,
	0.49628357344345131,
	-0.061080010938055818,
	0.6848195691385276,
}
@(private = "file")
NF_MAMBA_DTHETA_A_10 :: [?]f64 {
	-0.011938790296169996,
	-0.030976931633118297,
	-0.036860713845250012,
	0.018513459966433311,
	0.01597206780997534,
	-0.0022128593378983722,
	-0.029522337975121934,
	-0.0011678858630216268,
	0.028226203852116584,
	-0.054066287019770981,
	-0.044491077236726961,
	-0.0051236932602707532,
}
@(private = "file")
NF_OPT_REF :: [3][5]f64 {
	{
		0.11291540668959373,
		-0.60393906162792688,
		1.6015049004399815,
		1.3002390661235703,
		-0.50069442073052517,
	},
	{
		0.10021054778864248,
		-0.57104759816241801,
		1.529032716375093,
		1.2714370012167395,
		-0.48784063905080438,
	},
	{
		0.2100404701427469,
		0.0044957115663580849,
		1.3461376805709877,
		1.3495874999999999,
		-0.45718749999999997,
	},
}

@(private = "file")
NF_TOK_TEXTS :: [3]string {
	"Hello world!",
	"The Fed raised rates by 25bp.",
	"Stocks rallied; bonds fell.",
}
// Hugging Face transformers 4.24, add_special_tokens = True: GPT2Tokenizer
// (adds no special tokens) and BertTokenizer (bert-base-uncased, max_length 12)
@(private = "file")
NF_TOK_GPT2 :: [3][]int {
	{15496, 995, 0},
	{464, 10169, 4376, 3965, 416, 1679, 46583, 13},
	{1273, 3320, 32537, 26, 13100, 3214, 13},
}
@(private = "file")
NF_TOK_BERT :: [3][]int {
	{101, 7592, 2088, 999, 102, 0, 0, 0, 0, 0, 0, 0},
	{101, 1996, 7349, 2992, 6165, 2011, 2423, 2497, 2361, 1012, 102, 0},
	{101, 15768, 24356, 1025, 9547, 3062, 1012, 102, 0, 0, 0, 0},
}

// Deterministic parameter values (also in the torch reference): element j of
// parameter k is 0.5·sin(1 + 0.37 j + 1.1 k)
@(private = "file")
nf_fill_params :: proc(params: []^t.Tensor) {
	for p, k in params {
		for &v, j in p.data.data {v = 0.5 * math.sin(1.0 + 0.37 * f64(j) + 1.1 * f64(k))}
	}
}

// Mamba input [2, 5, 4]: scale · sin(0.9 i + 0.2) · (1 + 0.5 cos(0.3 i))
@(private = "file")
nf_mamba_input :: proc(scale: f64) -> ^t.Tensor {
	m := l.matrix_new(f64, 1, 40)
	for &v, i in m.data {
		v = scale * math.sin(0.9 * f64(i) + 0.2) * (1 + 0.5 * math.cos(0.3 * f64(i)))
	}
	x := t.tensor_new(m, false)
	x.shape = {2, 5, 4, 1}
	return x
}

@(private = "file")
nf_max_diff :: proc(a, b: []f64) -> f64 {
	if len(a) != len(b) {return math.INF_F64}
	d := 0.0
	for v, i in a {d = max(d, abs(v - b[i]))}
	return d
}

// Little-endian i32 at data[at:]
@(private = "file")
nf_i32 :: proc(data: []u8, at: int) -> int {
	x := u32(data[at]) | u32(data[at + 1]) << 8 | u32(data[at + 2]) << 16 | u32(data[at + 3]) << 24
	return int(i32(x))
}

@(private = "file")
nf_put_i32 :: proc(buf: ^[dynamic]u8, v: int) {
	x := u32(i32(v))
	append(buf, u8(x), u8(x >> 8), u8(x >> 16), u8(x >> 24))
}

// Bytes of save_checkpoint's optimizer block after the frozen flags
// (timestep, hyperparameters, moment count, moments, steps)
@(private = "file")
nf_adam_tail :: proc(opt: ^nn.Adam) -> int {
	n := 8 + 5 * 8 + 4
	for m in opt.moment_1 {n += 2 * (4 + 8 * len(m))}
	return n + 8 * len(opt.parameters)
}

// The file as written before the frozen flags existed (optimizer type 2): the
// type and flags of a type-3 file with n_all parameters replaced; `cut` bytes
// before the type are left out as well (an older layer layout)
@(private = "file")
nf_as_type2 :: proc(data: []u8, opt: ^nn.Adam, n_all, cut: int) -> (old: [dynamic]u8, ok: bool) {
	flags := 4 + 4 * n_all
	type_at := len(data) - nf_adam_tail(opt) - flags - 4
	if type_at < cut || nf_i32(data, type_at) != 3 || nf_i32(data, type_at + 4) != n_all {return}
	append(&old, ..data[:type_at - cut])
	nf_put_i32(&old, 2)
	append(&old, ..data[type_at + 4 + flags:])
	return old, true
}

// One Adam step on mean(out²) (the frozen-model checkpoint check), in a graph
// scope: the frozen first layer's outputs are not part of the graph
@(private = "file")
nf_frozen_step :: proc(model: ^nn.Sequential, opt: ^nn.Adam, x: ^t.Tensor) {
	scope: t.Graph_Scope
	defer t.graph_scope_destroy(&scope)
	nn.adam_zero_grad(opt)
	t.graph_begin(&scope)
	out := nn.sequential_forward(model, x)
	loss := t.tensor_mean(t.tensor_mul(out, out))
	t.tensor_backward(loss)
	t.graph_end(&scope)
	nn.adam_step(opt)
}

// torch 2.1.0+cpu nn.GRU(3, 4, batch_first), float64, seed 7: nonzero bias_hh and h_0
@(private = "file")
NF_GRU_X := [?]f64 {
	-0.042049767265182066,
	0.6628209471062041,
	1.3994091526870451,
	-1.4549862186629452,
	-0.9612955140847367,
	0.4375438588630386,
	1.7885417734026923,
	-0.08039804075733713,
	-1.5582993556390743,
	0.23963139750598528,
	1.7166036202837553,
	2.304738112599663,
	0.6534399108850214,
	0.333591585431449,
	1.1951147337146277,
	-0.4591714774219809,
	-1.718295141861475,
	1.530735153115545,
	0.9817931121202057,
	0.9570286079441692,
	1.1115773132932836,
	-0.30628264545022216,
	0.006669130470473673,
	1.760659506097817,
	1.2669030994331536,
	-0.2611596401485409,
	0.6226979404162903,
	-0.2220109975424316,
	0.08503882653303346,
	1.8639144412698772,
}
@(private = "file")
NF_GRU_H0 := [?]f64 {
	-0.9343354525485523,
	0.5913045579124961,
	-0.3063574952794949,
	-0.6300556586565216,
	-0.5738867238126416,
	-0.6354986068507187,
	0.4322423841080601,
	0.6747196427348519,
}
@(private = "file")
NF_GRU_R := [?]f64 {
	0.4473031392366413,
	-0.07113676313790206,
	1.5370663303415584,
	-0.40052933392917733,
	-0.16209438806403706,
	1.3801062866246665,
	-0.6574601912545139,
	-0.3701440301351539,
	1.7084188400263678,
	-1.2737445330897261,
	-0.23622143584532787,
	0.13363296079644807,
	-0.3648695566463679,
	-0.009535434337508955,
	1.4261135794673707,
	-0.6023430548019713,
	-1.2183680495720037,
	-0.1306189318118778,
	-0.7785942402415482,
	0.7932068600843698,
	-1.3839727713862389,
	-0.5030873419579612,
	1.6349644065547178,
	-0.42090529043170005,
	-1.027987348111354,
	-0.13056483319391798,
	0.47873679245305145,
	-0.22054992437304183,
	-0.9621190083162122,
	2.159574797483172,
	1.2661520710449452,
	0.9085380295939818,
	-0.3468836945295964,
	0.6734431995848988,
	0.5630489158138265,
	0.7187863937323853,
	-1.3007230746880825,
	-1.8473620852928343,
	-1.0017227114849618,
	-0.47827233682150283,
}
@(private = "file")
NF_GRU_W_IH := [?]f64 {
	0.22378830253547388,
	0.4076533022320815,
	0.35895795216914883,
	0.03634852358943097,
	-0.28725921041793034,
	-0.2165481595746262,
	-0.25615594502672434,
	0.5640783705073288,
	-0.0902467362340098,
	0.2625968501784921,
	-0.5902093166190224,
	0.3377537715448994,
	0.5066989902081119,
	0.43173333242163825,
	0.12908360709652644,
	0.16069596714793877,
	-0.049486857659715056,
	-0.1868386238670514,
	-0.3400588442239651,
	-0.21325568249590598,
	-0.2905118260478824,
	0.3013335341730198,
	0.4259834919007467,
	-0.3986674134204081,
	0.5439499508865225,
	0.2839500827833906,
	0.36907605385160525,
	0.03190204727072288,
	0.03000621370314782,
	0.017512053999478083,
	-0.23633033815671783,
	0.07537050357255393,
	0.5362363536850777,
	0.5951642235394755,
	-0.15499374446745945,
	-0.3196369880246483,
}
@(private = "file")
NF_GRU_W_HH := [?]f64 {
	-0.14269050511867654,
	-0.11859212889253606,
	-0.17012913509525962,
	0.2098946423919812,
	0.4489698931390892,
	0.003564800029427939,
	0.2065793533635636,
	0.5476818734736709,
	-0.3784898648319287,
	0.42786629707892077,
	-0.21721828806736693,
	-0.29650845910669577,
	-0.23993154319482463,
	-0.16115331128346344,
	-0.46027471116195906,
	0.5273929273093455,
	0.04196406177578227,
	0.1955802228461686,
	0.3904633632653549,
	-0.23193866839519012,
	-0.5911865136487663,
	-0.4123480667546646,
	-0.019802293129503545,
	0.5849120879284778,
	-0.2842807545130134,
	0.31972927008006236,
	0.3938836074926698,
	0.002258913688246822,
	0.15531702714486384,
	0.44817799375320544,
	-0.0638466955246838,
	0.4956530824309465,
	-0.4967087813770705,
	-0.496720797985563,
	-0.5931738147219923,
	-0.27974820035945636,
	-0.14916321607379485,
	0.4945002603319041,
	-0.26322871796275643,
	-0.2708175421297074,
	0.22318976898282394,
	0.2574070379153677,
	0.5064151272854288,
	-0.21769671339992935,
	0.5577302287557603,
	0.2179202631176851,
	0.056825998052748794,
	0.1534024228464559,
}
@(private = "file")
NF_GRU_BIAS := [?]f64 {
	0.4203729110947009,
	0.08789306955618326,
	0.777294411944055,
	-0.06567519845765302,
	0.4359464710777683,
	0.24100421824349116,
	-0.5426348718750686,
	0.1869992412571032,
	0.3928754645216125,
	0.10906669400523583,
	-0.241871660255507,
	-0.42247180978943033,
}
@(private = "file")
NF_GRU_BIAS_HN := [?]f64 {
	0.3098378999026229,
	0.3633210476606885,
	0.20118431380245838,
	-0.4183311423867313,
}
@(private = "file")
NF_GRU_OUT := [?]f64 {
	-0.022174893179615296,
	0.6723914820474786,
	0.14706664854764062,
	-0.7217536276140363,
	0.190430500898005,
	0.353381478698383,
	0.16083664308189538,
	-0.6774093105205234,
	-0.2625464653530671,
	-0.01828162619794571,
	-0.4927578569302752,
	-0.44726872637875176,
	0.30495018364956805,
	0.5886997097532618,
	0.31073198152793574,
	-0.7228901069257903,
	0.31020130486579506,
	0.6800744396812229,
	-0.4178252944719469,
	-0.7280209382023217,
	-0.03382656829123187,
	-0.31264142716613874,
	-0.1990943832394218,
	0.17439793203559262,
	0.3781355790569695,
	0.39117325950269977,
	-0.27075758128683525,
	-0.18157547726846013,
	0.5439479401110198,
	0.600733555802659,
	-0.18145676670400554,
	-0.48557057650780333,
	0.4675018157234421,
	0.6645521900321189,
	-0.6061273306534833,
	-0.42839828542251923,
	0.6002301664805615,
	0.7753187913283157,
	-0.23534257497295413,
	-0.6182515047668361,
}
@(private = "file")
NF_GRU_DX := [?]f64 {
	-0.48624510424460865,
	0.543264937341112,
	0.07384432814455794,
	0.19357361057984046,
	-0.08318634013766218,
	0.2551476958311778,
	-0.11470023443023936,
	-0.30273186049742473,
	-0.161862010321336,
	-0.7047053125803386,
	0.5780637116048978,
	-0.15071331312247385,
	0.3272058423222061,
	0.006545920784513451,
	-0.1745993752368191,
	-0.6824218309618997,
	0.07965949938554477,
	-0.14216791080753918,
	-0.17185225703265952,
	0.25453384638751814,
	-0.17330878709566616,
	-0.28617625724538603,
	0.5004616579285351,
	-0.0645936154784589,
	-0.04557491349496749,
	0.1274159634416143,
	-0.2788354295962278,
	0.2014488952107556,
	-0.3478800125854742,
	-0.10344473464586705,
}
@(private = "file")
NF_GRU_DH0 := [?]f64 {
	-0.027532652628416865,
	-0.06347944004895698,
	-0.2746430010715902,
	-0.24995896349314817,
	-1.0862246568304523,
	0.2296250467982614,
	0.5390718041227364,
	0.24502149894398656,
}
@(private = "file")
NF_GRU_DW_IH := [?]f64 {
	0.006656898119094076,
	-0.010507530425880271,
	0.039909790412526396,
	-0.029963459919607728,
	0.49123098681027755,
	-0.8918746970381826,
	-0.4272407259018289,
	0.0331460555894392,
	-0.4572066151717432,
	-1.4643917797391977,
	0.5061560407134469,
	0.47319738096239383,
	0.008408113149728957,
	0.007302910830661514,
	0.1782660686573723,
	-0.03624911944542754,
	-0.5891397752698434,
	-0.5007551456804545,
	-1.1489539857452622,
	0.1359764637661103,
	-0.39257612406238773,
	-0.06195792574570315,
	2.337037559237706,
	0.23545207195480955,
	0.11733585672997142,
	0.02240408909771046,
	0.3013770590808649,
	-0.022510347879742463,
	1.9370595462893347,
	-0.3364436273053866,
	0.3681905752066096,
	0.07713424719994662,
	-2.0804197082348637,
	1.348547424776628,
	5.38088612391388,
	-0.050090111365428125,
}
@(private = "file")
NF_GRU_DW_HH := [?]f64 {
	0.0351959435863027,
	0.0023972787287664028,
	-0.11366220924989824,
	-0.009844367084863033,
	-0.10185016245849099,
	-0.059484160517627116,
	-0.07152383521614873,
	0.11577171206668088,
	-0.3878709138662193,
	-0.11541043318065855,
	-1.2148942875318247,
	0.1629420437680293,
	0.05361691692681473,
	-0.01841673572632649,
	0.023150132807843733,
	-0.0167901300931926,
	-0.6297654976474663,
	0.10237177230133035,
	-0.35686779490000703,
	0.059873166013555024,
	-0.2525617536852505,
	-0.08339170624783729,
	-0.201497371733855,
	-0.01585817803595481,
	0.003824536030688209,
	0.0043402276552678455,
	-0.03809298542250267,
	0.009346632072948065,
	0.13052080913461545,
	0.08431084891881689,
	0.1704504087373655,
	-0.043958189684779825,
	0.0967835170388296,
	-0.052763113045694796,
	-0.6955161447312913,
	0.013226195344603701,
	-0.040652261544973985,
	0.022028952932800386,
	-0.06345489157478001,
	0.01610619044777698,
	0.4680883937513798,
	-0.015021633720650188,
	0.5316670001547303,
	-0.030405623724329846,
	0.2344400444955918,
	0.1799230309049144,
	-0.19167098983147957,
	0.030149933840126028,
}
@(private = "file")
NF_GRU_DBIAS := [?]f64 {
	0.02663907553861166,
	-0.011353567388543505,
	0.1794093125919875,
	-0.016227500007294986,
	1.5838639913208155,
	-0.595073027749117,
	0.0868406940880463,
	0.07786623258199799,
	-0.9420504874957218,
	-0.2221803150043382,
	2.6009170909911337,
	-0.06436431155033123,
}
@(private = "file")
NF_GRU_DBIAS_HN := [?]f64 {
	-0.8439471432843575,
	-0.13184069850087643,
	2.2482297659814545,
	-0.006867440711646994,
}

// torch 2.1.0+cpu gelu(approximate="tanh"), float64
@(private = "file")
NF_GT_X := [?]f64{-6.0, -3.1, -1.2, -0.45, 0.0, 0.3, 0.9, 2.2, 5.5, 12.0}
@(private = "file")
NF_GT_W := [?]f64{0.7, -1.1, 0.4, 2.0, -0.3, 1.3, -0.8, 0.6, 1.7, -0.5}
@(private = "file")
NF_GT_Y := [?]f64 {
	-8.43964897967453e-11,
	-0.0026266102305497474,
	-0.13829723086213508,
	-0.14687154361411522,
	0.0,
	0.18537092354275922,
	0.7342284887097207,
	2.1696785270380117,
	5.499999994072359,
	12.0,
}
@(private = "file")
NF_GT_DX := [?]f64 {
	-5.396983209062729e-10,
	0.009613335656392507,
	-0.04709045389324504,
	0.3284288856530348,
	-0.15,
	0.9519840871304424,
	-0.8440537638076933,
	0.6389513695473311,
	1.7000000795013976,
	-0.5,
}

nn_fixes_test :: proc(allocator: mem.Allocator) {
	seed_test(#procedure)
	fmt.println("\n=== nn regression tests ===")
	passed, total := 0, 0
	report :: proc(name: string, ok: bool, passed, total: ^int) {
		total^ += 1
		if ok {passed^ += 1}
		fmt.printf("%s %s\n", "OK  " if ok else "FAIL", name)
	}
	flat :: proc(c: [4]int) -> bool {return c[0] == c[1] && c[1] == c[2] && c[2] == c[3]}

	track: mem.Tracking_Allocator
	mem.tracking_allocator_init(&track, runtime.default_allocator())
	defer mem.tracking_allocator_destroy(&track)
	context.allocator = mem.tracking_allocator(&track)

	nf_ids_t = nf_ids({1, 4, 2, 7, 0, 0, 3, 5, 9, 2, 1, 6}, 2, 6)
	nf_seg_t = nf_ids({0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1}, 2, 6)
	nf_x = nf_seq(2, 5, 4)

	{ 	// GPT and BERT training steps in a graph scope (position ids, [CLS] ids)
		nf_gpt = nn.gpt_model_new(11, 8, 2, 16, 2, 6)
		nf_bert = nn.bert_model_new(11, 8, 2, 16, 2, 6)
		g := nf_steps(&track, true, proc() -> ^t.Tensor {
			logits := nn.gpt_model_forward(&nf_gpt, nf_ids_t, nil, true)
			return t.tensor_cross_entropy_loss(nn.as_matrix(logits, 12, 11), NF_TARGETS)
		})
		b := nf_steps(&track, true, proc() -> ^t.Tensor {
			mlm, nsp := nn.bert_model_forward(&nf_bert, nf_ids_t, nf_seg_t, true)
			return t.tensor_add(
				t.tensor_cross_entropy_loss(nn.as_matrix(mlm, 12, 11), NF_TARGETS),
				t.tensor_cross_entropy_loss(nsp, {0, 1}),
			)
		})
		report(
			fmt.tprintf("GPT / BERT steps in a graph scope: no heap growth %v %v", g, b),
			flat(g) && flat(b),
			&passed,
			&total,
		)
		nn.gpt_model_free(&nf_gpt)
		nn.bert_model_free(&nf_bert)
	}
	{ 	// Sequential LSTM / GRU / Mamba zero states in a graph scope
		nf_model = nn.sequential_new()
		nn.sequential_add(
			nf_model,
			nn.lstm_layer_new(4, 3),
			nn.gru_layer_new(3, 3),
			nn.linear_layer_new(3, 4),
			nn.mamba_layer_new(4, 3),
		)
		c := nf_steps(&track, true, proc() -> ^t.Tensor {
				out := nn.sequential_forward(nf_model, nf_x)
				return t.tensor_mean(t.tensor_mul(out, out))
			})
		report(
			fmt.tprintf("Sequential LSTM, GRU, Mamba in a graph scope: no heap growth %v", c),
			flat(c),
			&passed,
			&total,
		)
		nn.sequential_free(nf_model)
	}
	{ 	// MoE first layer, no aux loss, tensor_free_graph (no scope)
		nf_model = nn.sequential_new()
		nn.sequential_add(nf_model, nn.moe_layer_new(4, 6, 3, 2))
		c := nf_steps(&track, false, proc() -> ^t.Tensor {
			out := nn.sequential_forward(nf_model, nf_x)
			return t.tensor_mean(t.tensor_mul(out, out))
		})
		report(
			fmt.tprintf("MoE without aux loss, tensor_free_graph: no heap growth %v", c),
			flat(c),
			&passed,
			&total,
		)
		nn.sequential_free(nf_model)
	}
	{ 	// losses with data inputs: fixed variance, fixed scale and ν, data predictions
		m := l.matrix_new(f64, 4, 1) // mu is learned, the rest is data
		copy(m.data, []f64{0.1, -0.2, 0.3, 0.0})
		mu := t.tensor_new(m, true)
		nf_const := [3]^t.Tensor{}
		for &c, k in nf_const {
			cm := l.matrix_new(f64, 4, 1)
			for &v in cm.data {v = 1.5 + f64(k)}
			c = t.tensor_new(cm, false)
		}
		counts: [4]int
		for i in 0 ..< 4 {
			g := nn.gaussian_nll_loss(mu, nf_const[0], nf_const[1])
			t.tensor_backward(g)
			t.tensor_free_graph(g)
			s := nn.student_t_nll_loss(mu, nf_const[0], nf_const[2], nf_const[1])
			t.tensor_backward(s)
			t.tensor_free_graph(s)
			q := nn.quantile_loss(nf_const[0], {0.5, 1.0, 2.0, 0.0}, {0.5}) // no gradient at all
			t.tensor_free(q)
			counts[i] = len(track.allocation_map)
		}
		report(
			fmt.tprintf("losses with data var / sigma / nu / pred: no heap growth %v", counts),
			flat(counts),
			&passed,
			&total,
		)
		t.tensor_free(mu)
		for c in nf_const {t.tensor_free(c)}
	}
	{ 	// a layer header with 0 attention heads is a corrupt file
		path := "model_checkpoints/nn_fixes_heads0.ckpt"
		os.make_directory("model_checkpoints")
		s := nn.sequential_new()
		nn.sequential_add(s, nn.multi_head_attention_layer_new(4, 2), nn.linear_layer_new(4, 2))
		opt := nn.adam_new()
		nn.sequential_add_to_opt(s, &opt)
		saved := nn.save_checkpoint(s, &opt, path, 1)
		data, _ := os.read_entire_file(path, context.allocator)
		// magic (10 bytes), epoch, layer count, layer id, d_model, then num_heads
		heads_at := len(nn.CHECKPOINT_MAGIC) + 4 * 4
		good := saved && data[heads_at] == 2
		for i in 0 ..< 4 {data[heads_at + i] = 0}
		_ = os.write_entire_file(path, data)
		before := len(track.allocation_map)
		m, o, _, ok := nn.load_checkpoint(path)
		good = good && !ok && m == nil && o == nil && len(track.allocation_map) == before
		_, _, pok := nn.sequential_load_partial(s, path)
		good = good && !pok
		report(
			"checkpoint with 0 heads: load_checkpoint and load_partial fail, no leak",
			good,
			&passed,
			&total,
		)
		delete(data)
		_ = os.remove(path)
		nn.adam_free(&opt)
		nn.sequential_free(s)
	}

	{ 	// WordPiece drops private-use and unassigned characters, as BertTokenizer
		path := "model_checkpoints/nn_fixes_vocab.txt"
		_ = os.write_entire_file(
			path,
			transmute([]u8)string("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nprivate\nuse\n"),
		)
		wp, ok := tok.wordpiece_tokenizer_new(path, 8)
		ids := tok.wordpiece_encode(&wp, "private  use ͸") if ok else nil
		good := ok && len(ids) == 2 && ids[0] == 5 && ids[1] == 6 // was 5, [UNK], 6, [UNK]
		report(
			fmt.tprintf("WordPiece: private-use and unassigned characters dropped %v", ids),
			good,
			&passed,
			&total,
		)
		delete(ids)
		if ok {tok.wordpiece_tokenizer_free(&wp)}
		_ = os.remove(path)
	}

	{ 	// bpe_tokenize adds [CLS] / [SEP] only where the vocabulary has them
		vocab, merges :=
			"model_checkpoints/nn_fixes_bpe.json", "model_checkpoints/nn_fixes_bpe.txt"
		_ = os.write_entire_file(
			vocab,
			transmute([]u8)string(`{"!": 0, "a": 1, "b": 2, "ab": 3, "<|endoftext|>": 4}`),
		)
		_ = os.write_entire_file(merges, transmute([]u8)string("#version: 0.2\na b\n"))
		bt, ok := tok.bpe_tokenizer_new(vocab, merges, 6)
		good := ok && bt.cls_id == -1 && bt.sep_id == -1
		if ok {
			ids, mask := tok.bpe_tokenize(&bt, "aba") // default add_special_tokens = true
			good &&=
				slice.equal(ids, []int{3, 1, 0, 0, 0, 0}) &&
				slice.equal(mask, []int{1, 1, 0, 0, 0, 0})
			delete(ids)
			delete(mask)
			tok.bpe_tokenizer_free(&bt)
		}
		_ = os.remove(vocab)
		_ = os.remove(merges)
		report(
			"bpe_tokenize on a vocabulary without [CLS] / [SEP] adds no special tokens",
			good,
			&passed,
			&total,
		)

		// the real vocabularies vs Hugging Face (where downloaded, as tokenizer_reference_test)
		GPT2_VOCAB, GPT2_MERGES, BERT_VOCAB ::
			"model_checkpoints/vocab.json",
			"model_checkpoints/merges.txt",
			"model_checkpoints/bert_vocab.txt"
		if !os.exists(GPT2_VOCAB) || !os.exists(GPT2_MERGES) || !os.exists(BERT_VOCAB) {
			fmt.println(
				"SKIP GPT-2 / BERT vocabularies not downloaded (see tokenizer_reference_test)",
			)
		} else {
			gpt2, g_ok := tok.bpe_tokenizer_new(GPT2_VOCAB, GPT2_MERGES, 12)
			bert, b_ok := tok.wordpiece_tokenizer_new(BERT_VOCAB, 12)
			same := g_ok && b_ok
			ref_gpt2, ref_bert := NF_TOK_GPT2, NF_TOK_BERT
			for text, i in NF_TOK_TEXTS {
				if !same {break}
				ids, mask := tok.bpe_tokenize(&gpt2, text)
				want := ref_gpt2[i]
				n := 0
				for m in mask {n += m}
				same &&= n == len(want) && slice.equal(ids[:n], want)
				delete(ids)
				delete(mask)
				ids, mask = tok.tokenize(&bert, text)
				same &&= slice.equal(ids, ref_bert[i])
				delete(ids)
				delete(mask)
			}
			if g_ok {tok.bpe_tokenizer_free(&gpt2)}
			if b_ok {tok.wordpiece_tokenizer_free(&bert)}
			report(
				"default tokenization of 3 texts as Hugging Face: GPT-2 without special tokens, BERT [CLS] … [SEP]",
				same,
				&passed,
				&total,
			)
		}
	}

	{ 	// FlashAttention GPT honours an explicit mask (a causal one keeps the kernel)
		nf_gpt = nn.gpt_model_new(11, 8, 2, 16, 2, 6, use_flash = true)
		causal := nn.create_causal_mask(6, context.allocator)
		custom := nn.create_causal_mask(6, context.allocator)
		for i in 1 ..< 6 {custom[i * 6] = -1e9} 	// the first token hidden from the others
		logits :: proc(mask: []f64, flash: bool) -> []f64 {
			for &b in nf_gpt.blocks {b.mha.use_flash = flash}
			out := nn.gpt_model_forward(&nf_gpt, nf_ids_t, mask, false)
			vals := slice.clone(out.data.data)
			t.tensor_free_graph(out)
			return vals
		}
		flash_custom, exact_custom := logits(custom, true), logits(custom, false)
		flash_causal, exact_causal := logits(causal, true), logits(causal, false)
		flash_nil := logits(nil, true)
		d_custom := nf_max_diff(flash_custom, exact_custom)
		d_effect := nf_max_diff(flash_custom, flash_causal)
		d_causal := nf_max_diff(flash_causal, exact_causal)
		d_nil := nf_max_diff(flash_causal, flash_nil)
		report(
			fmt.tprintf(
				"GPT use_flash with a custom mask = exact path %.1e (the mask moves the logits by %.2f); causal mask: flash vs exact %.1e, vs no mask %.1e",
				d_custom,
				d_effect,
				d_causal,
				d_nil,
			),
			d_custom < 1e-12 && d_effect > 1e-3 && d_causal < 1e-12 && d_nil == 0,
			&passed,
			&total,
		)
		for v in ([][]f64{flash_custom, exact_custom, flash_causal, exact_causal, flash_nil, causal, custom}) {delete(v)}
		nn.gpt_model_free(&nf_gpt)
	}

	{ 	// zero_grad drops the gradients and a parameter without one is skipped:
		// torch 2.1 Adam(lr 0.1), AdamW(lr 0.1, wd 0.1), SGD(lr 0.1, momentum
		// 0.9, wd 0.05), the second parameter not in the loss of steps 1 and 2
		worst := 0.0
		dropped := true
		refs := NF_OPT_REF
		tensor :: proc(vals: []f64, rg: bool) -> ^t.Tensor {
			m := l.matrix_new(f64, 1, len(vals))
			copy(m.data, vals)
			return t.tensor_new(m, rg)
		}
		for kind in 0 ..< 3 {
			p1, p2 := tensor({0.5, -1.0, 2.0}, true), tensor({1.5, -0.7}, true)
			c1, c2 := tensor({1.0, 2.0, 0.5}, false), tensor({0.3, 1.2}, false)
			adam: nn.Adam
			sgd: nn.SGD
			if kind < 2 {
				adam = nn.adam_new(0.1, weight_decay = 0.1 if kind == 1 else 0)
				nn.adam_add_param(&adam, p1)
				nn.adam_add_param(&adam, p2)
			} else {
				sgd = nn.sgd_new(0.1, momentum = 0.9, weight_decay = 0.05)
				nn.sgd_add_param(&sgd, p1)
				nn.sgd_add_param(&sgd, p2)
			}
			buf1: [^]f64
			for k in 0 ..< 4 {
				if kind < 2 {nn.adam_zero_grad(&adam)} else {nn.sgd_zero_grad(&sgd)}
				dropped &&= !t.tensor_has_grad(p1) && !t.tensor_has_grad(p2)
				loss := t.tensor_mean(t.tensor_mul(t.tensor_mul(p1, p1), c1))
				if k != 1 && k != 2 {
					loss = t.tensor_add(
						loss,
						t.tensor_mean(t.tensor_mul(t.tensor_mul(p2, p2), c2)),
					)
				}
				t.tensor_backward(loss)
				dropped &&= t.tensor_has_grad(p1) && !t.tensor_has_grad(p2) == (k == 1 || k == 2)
				// the buffers are reused, not reallocated, from step to step
				if k ==
				   0 {buf1 = raw_data(p1.grad.data)} else {dropped &&= raw_data(p1.grad.data) == buf1}
				if kind < 2 {nn.adam_step(&adam)} else {nn.sgd_step(&sgd)}
				t.tensor_free_graph(loss)
			}
			got := [5]f64 {
				p1.data.data[0],
				p1.data.data[1],
				p1.data.data[2],
				p2.data.data[0],
				p2.data.data[1],
			}
			ref := refs[kind]
			worst = max(worst, nf_max_diff(got[:], ref[:]))
			if kind < 2 {nn.adam_free(&adam)} else {nn.sgd_free(&sgd)}
			for x in ([]^t.Tensor{p1, p2, c1, c2}) {t.tensor_free(x)}
		}
		report(
			fmt.tprintf(
				"adam / sgd_zero_grad drop the gradients %v; Adam, AdamW, SGD momentum skip a parameter without one, vs torch 2.1: %.1e",
				dropped,
				worst,
			),
			dropped && worst < 1e-14,
			&passed,
			&total,
		)
	}

	{ 	// checkpoints keep frozen parameters frozen, with the Adam state intact
		path := "model_checkpoints/nn_fixes_frozen.ckpt"
		model := nn.sequential_new()
		nn.sequential_add(
			model,
			nn.linear_layer_new(3, 4),
			nn.Activation.Tanh,
			nn.linear_layer_new(4, 2),
		)
		nn.sequential_freeze_range(model, 0, 1)
		opt := nn.adam_new(0.05, weight_decay = 0.01)
		nn.sequential_add_to_opt(model, &opt)
		xm := l.matrix_new(f64, 5, 3)
		for &v, i in xm.data {v = math.sin(0.7 * f64(i))}
		x := t.tensor_new(xm, false)
		for _ in 0 ..< 3 {nf_frozen_step(model, &opt, x)}
		saved := nn.save_checkpoint(model, &opt, path, 3)
		m2, o2, epoch, ok := nn.load_checkpoint(path)
		good := saved && ok && epoch == 3
		if ok {
			all1 := nn.sequential_parameters(model, all = true)
			all2 := nn.sequential_parameters(m2, all = true)
			good &&= len(all1) == len(all2) && len(o2.parameters) == len(opt.parameters)
			for p, i in all1 {good &&= good && p.requires_grad == all2[i].requires_grad}
			for i in 0 ..< len(opt.parameters) {
				if !good {break}
				good &&=
					slice.equal(opt.moment_1[i], o2.moment_1[i]) &&
					slice.equal(opt.moment_2[i], o2.moment_2[i]) &&
					opt.steps[i] == o2.steps[i]
			}
			good &&= opt.timestep == o2.timestep
			m2.training = model.training
			nf_frozen_step(model, &opt, x) // the next step: the same in both
			nf_frozen_step(m2, o2, x)
			for p, i in all1 {good &&= good && slice.equal(p.data.data, all2[i].data.data)}
			delete(all1)
			delete(all2)
			nn.sequential_free(m2)
			nn.adam_free(o2)
			free(o2)
		}
		// a checkpoint without the frozen flags (optimizer type 2) still loads, all trainable
		data, _ := os.read_entire_file(path, context.allocator)
		old, old_ok := nf_as_type2(data, &opt, 4, 0)
		_ = os.write_entire_file(path, old[:])
		m3, o3, _, ok3 := nn.load_checkpoint(path)
		old_ok &&= ok3
		if ok3 {
			all3 := nn.sequential_parameters(m3, all = true)
			for p in all3 {old_ok &&= p.requires_grad}
			delete(all3)
			nn.sequential_free(m3)
			nn.adam_free(o3)
			free(o3)
		}
		delete(old)
		delete(data)
		_ = os.remove(path)
		report(
			fmt.tprintf(
				"partly frozen model with Adam state: save → load keeps the frozen flags, moments and steps, identical next step %v; a file without the flags loads %v",
				good,
				old_ok,
			),
			good && old_ok,
			&passed,
			&total,
		)
		t.tensor_free(x)
		nn.adam_free(&opt)
		nn.sequential_free(model)
	}

	{ 	// Mamba: pre-norm residual block vs a torch reimplementation (the
		// state-spaces/mamba Block, RMSNorm + residual, around wotan's mixer)
		mamba := nn.mamba_layer_new(4, 3)
		ps := make([dynamic]^t.Tensor)
		layer := nn.Layer(mamba)
		nn.layer_parameters(&layer, &ps, all = true)
		nf_fill_params(ps[:])
		d_out, d_grad := 0.0, 0.0
		mixer: [2][]f64
		for scale, s in ([]f64{1, 10}) {
			x := nf_mamba_input(scale)
			h0 := t.tensor_new(l.matrix_new(f64, 1, 2 * 4 * 3), false)
			h0.shape = {2, 4, 3, 1}
			out := nn.mamba_layer_forward(&mamba, x, h0)
			for p in ps {t.tensor_zero_grad(p)}
			loss := t.tensor_mean(t.tensor_mul(out, out))
			t.tensor_backward(loss)
			ref_out := NF_MAMBA_OUT_1 if s == 0 else NF_MAMBA_OUT_10
			ref_dg := NF_MAMBA_DGAMMA_1 if s == 0 else NF_MAMBA_DGAMMA_10
			ref_da := NF_MAMBA_DTHETA_A_1 if s == 0 else NF_MAMBA_DTHETA_A_10
			d_out = max(d_out, nf_max_diff(out.data.data, ref_out[:]) / scale)
			d_grad = max(
				d_grad,
				nf_max_diff(mamba.norm.gamma.grad.data, ref_dg[:]),
				nf_max_diff(mamba.A.grad.data, ref_da[:]),
			)
			mixer[s] = make([]f64, 40)
			for v, i in out.data.data {mixer[s][i] = v - x.data.data[i]}
			t.tensor_free_graph(loss)
			t.tensor_free(x)
			t.tensor_free(h0)
		}
		d_scale := nf_max_diff(mixer[0], mixer[1])
		report(
			fmt.tprintf(
				"Mamba block (RMSNorm, residual) vs torch: output %.1e, grads of norm weight / θ_A %.1e; 10σ input: SSM part within %.1e of the 1σ one",
				d_out,
				d_grad,
				d_scale,
			),
			d_out < 1e-12 && d_grad < 1e-12 && d_scale < 1e-4,
			&passed,
			&total,
		)
		delete(mixer[0])
		delete(mixer[1])

		// checkpoints: the norm weight round-trips; a Mamba layer of an older
		// file (layer id 25: no norm, no residual) loads as a plain layer that
		// computes proj_out(SSM(x)): fed RMSNorm(x)·γ it gives torch's block
		// output minus x; saved again it keeps id 25
		path := "model_checkpoints/nn_fixes_mamba.ckpt"
		s := nn.sequential_new()
		nn.sequential_add(s, mamba)
		opt := nn.adam_new()
		saved := nn.save_checkpoint(s, &opt, path, 0)
		m2, o2, _, ok := nn.load_checkpoint(path)
		good := saved && ok
		if ok {
			p2 := nn.sequential_parameters(m2, all = true)
			good &&= len(p2) == len(ps)
			for p, i in ps {good &&= good && slice.equal(p.data.data, p2[i].data.data)}
			delete(p2)
			nn.sequential_free(m2)
			nn.adam_free(o2)
			free(o2)
		}
		data, _ := os.read_entire_file(path, context.allocator)
		id_at := len(nn.CHECKPOINT_MAGIC) + 8
		d_plain := -1.0
		good &&= nf_i32(data, id_at) == 34
		data[id_at] = 25 // the older layer id, without the norm weight (8 + 8·4 bytes)
		old, old_ok := nf_as_type2(data, &opt, len(ps), 8 + 8 * 4)
		_ = os.write_entire_file(path, old[:])
		m3, o3, _, ok3 := nn.load_checkpoint(path)
		old_ok &&= ok3
		if ok3 {
			p3 := nn.sequential_parameters(m3, all = true)
			old_ok &&= len(p3) == len(ps)
			if old_ok {
				for p, i in ps[1:] {old_ok &&= slice.equal(p.data.data, p3[i + 1].data.data)}
			}
			delete(p3)
			plain := &m3.layers[0].(nn.MambaLayer)
			old_ok &&= plain.plain
			x := nf_mamba_input(1)
			h0 := t.tensor_new(l.matrix_new(f64, 1, 2 * 4 * 3), false)
			h0.shape = {2, 4, 3, 1}
			xn := nn.layer_norm_layer_forward(&mamba.norm, x)
			out3 := nn.mamba_layer_forward(plain, xn, h0)
			d_plain = 0
			ref1 := NF_MAMBA_OUT_1
			for v, i in out3.data.data {d_plain = max(d_plain, abs(v - (ref1[i] - x.data.data[i])))}
			old_ok &&= d_plain < 1e-12
			t.tensor_free_graph(out3)
			t.tensor_free(x)
			t.tensor_free(h0)
			// saved again: still layer id 25, still plain
			opt3 := nn.adam_new()
			nn.save_checkpoint(m3, &opt3, path, 0)
			nn.adam_free(&opt3)
			again, _ := os.read_entire_file(path, context.allocator)
			old_ok &&= nf_i32(again, id_at) == 25
			delete(again)
			m4, o4, _, ok4 := nn.load_checkpoint(path)
			old_ok &&= ok4 && m4.layers[0].(nn.MambaLayer).plain
			if ok4 {nn.sequential_free(m4); nn.adam_free(o4); free(o4)}
			nn.sequential_free(m3)
			nn.adam_free(o3)
			free(o3)
		}
		delete(old)
		delete(data)
		_ = os.remove(path)
		report(
			fmt.tprintf(
				"Mamba checkpoint: norm weight saved and loaded %v; an older Mamba layer loads as a plain layer (its own function: vs torch %.1e, saved again as id 25) %v",
				good,
				d_plain,
				old_ok,
			),
			good && old_ok,
			&passed,
			&total,
		)
		delete(ps)
		nn.adam_free(&opt)
		nn.sequential_free(s) // frees the Mamba layer
	}

	{ 	// frozen leading layers without a graph scope: the forward passes free
		// their constant intermediates (or hand them to the graph)
		live :: proc() -> int {return t.global_tensors_created - t.global_tensors_freed}
		allocs := len(track.allocation_map)
		model := nn.sequential_new()
		nn.sequential_add(
			model,
			nn.linear_layer_new(4, 8),
			nn.Activation.ReLU,
			nn.linear_layer_new(8, 8),
			nn.Activation.ReLU,
			nn.linear_layer_new(8, 2),
		)
		nn.sequential_freeze_range(model, 0, 2)
		xm := l.matrix_new(f64, 1, 4)
		copy(xm.data, []f64{0.1, 0.2, 0.3, 0.4})
		x := t.tensor_new(xm, false)
		start := live()
		for _ in 0 ..< 3 {
			loss := t.tensor_cross_entropy_loss(nn.sequential_forward(model, x), {1})
			t.tensor_backward(loss)
			t.tensor_free_graph(loss)
		}
		partly := live() - start

		// fully frozen: the output is kept, everything before it is freed
		nn.sequential_freeze_all(model)
		out := nn.sequential_forward(model, x)
		out_ok := !out.requires_grad && len(out.data.data) == 2
		t.tensor_free(out)
		fully := live() - start

		// features of the frozen model, detached once and reused by a trainable head
		head := nn.sequential_new()
		nn.sequential_add(head, nn.linear_layer_new(2, 2))
		start_cached := live()
		o := nn.sequential_forward(model, x)
		feats := t.tensor_detach(o)
		t.tensor_free(o)
		snapshot := slice.clone(feats.data.data, context.temp_allocator)
		for _ in 0 ..< 3 {
			loss := t.tensor_cross_entropy_loss(nn.sequential_forward(head, feats), {1})
			t.tensor_backward(loss)
			t.tensor_free_graph(loss)
		}
		cached_ok := slice.equal(feats.data.data, snapshot)
		t.tensor_free(feats)
		cached := live() - start_cached

		// GPT with frozen embeddings and first block, no scope
		gpt := nn.gpt_model_new(11, 8, 2, 16, 2, 6)
		nn.tensor_set_requires_grad(gpt.token_emb.weight, false)
		nn.tensor_set_requires_grad(gpt.pos_emb.weight, false)
		nn.gpt_freeze_blocks(&gpt, 0, 1)
		start_gpt := live()
		for _ in 0 ..< 2 {
			logits := nn.gpt_model_forward(&gpt, nf_ids_t, nil, true)
			loss := t.tensor_cross_entropy_loss(nn.as_matrix(logits, 12, 11), NF_TARGETS)
			t.tensor_backward(loss)
			t.tensor_free_graph(loss)
		}
		gpt_left := live() - start_gpt
		nn.gpt_model_free(&gpt)
		nn.sequential_free(model)
		nn.sequential_free(head)
		t.tensor_free(x)
		leak := len(track.allocation_map) - allocs
		report(
			fmt.tprintf(
				"frozen leading layers without a scope: %d tensors left (Sequential), %d (GPT); fully frozen forward %d; detached features reused %d (intact %v); %d allocations left",
				partly,
				gpt_left,
				fully,
				cached,
				cached_ok,
				leak,
			),
			partly == 0 &&
			gpt_left == 0 &&
			fully == 0 &&
			out_ok &&
			cached == 0 &&
			cached_ok &&
			leak == 0,
			&passed,
			&total,
		)
	}

	{ 	// every layer and model forward records itself and leaves only its output
		// behind, which belongs to the caller: a frozen FFN on data leaves exactly its
		// output per step (freed by the caller, nothing is left); a frozen encoder
		// block chained by hand into a head, recorded with nn.forward_record_begin /
		// forward_record_end, leaves nothing; an MLP with frozen hidden layers leaves
		// nothing; two frozen linears chained by hand leave their two outputs per step
		// without the recording and nothing with it
		live :: proc() -> int {return t.global_tensors_created - t.global_tensors_freed}
		freeze :: proc(lay: nn.Layer) {
			lay := lay
			ps := make([dynamic]^t.Tensor)
			defer delete(ps)
			nn.layer_parameters(&lay, &ps, all = true)
			for p in ps {nn.tensor_set_requires_grad(p, false)}
		}
		allocs := len(track.allocation_map)
		xm := l.matrix_new(f64, 3, 4)
		for &v, i in xm.data {v = 0.1 * f64(i % 7) - 0.3}
		x := t.tensor_new(xm, false)
		xs := t.tensor_new(l.matrix_from_flat(xm.data, 1, 12), false)
		xs.shape = {1, 3, 4, 1}
		targets := []int{1, 0, 1}
		head := nn.linear_layer_new(4, 2)
		step :: proc(head: ^nn.LinearLayer, h: ^t.Tensor, targets: []int) {
			loss := t.tensor_cross_entropy_loss(nn.linear_forward(head, h), targets)
			t.tensor_backward(loss)
			t.tensor_free_graph(loss)
		}

		ffn := nn.ffn_layer_new(4, 8)
		freeze(ffn)
		s0 := live()
		outs: [3]^t.Tensor
		for &h in outs {
			h = nn.ffn_layer_forward(&ffn, x)
			step(&head, h, targets)
		}
		ffn_out_only := live() - s0 // the 3 outputs, nothing from inside the FFN
		for h in outs {t.tensor_free(h)} 	// constants: the caller's
		left_ffn := live() - s0

		enc := nn.transformer_encoder_block_new(4, 2, 8)
		freeze(enc)
		s1 := live()
		for _ in 0 ..< 3 {
			rec := nn.forward_record_begin()
			loss := t.tensor_cross_entropy_loss(
				nn.linear_forward(
					&head,
					nn.as_matrix(nn.transformer_encoder_block_forward(&enc, xs), 3, 4),
				),
				targets,
			)
			nn.forward_record_end(rec, loss)
			t.tensor_backward(loss)
			t.tensor_free_graph(loss)
		}
		left_enc := live() - s1

		mlp := nn.mlp_new([]int{4, 8, 8, 2})
		for i in 0 ..< 2 {freeze(mlp.layers[i])}
		s2 := live()
		for _ in 0 ..< 3 {
			loss := t.tensor_cross_entropy_loss(nn.mlp_forward(&mlp, x, 0, false), targets)
			t.tensor_backward(loss)
			t.tensor_free_graph(loss)
		}
		left_mlp := live() - s2

		lin1, lin2 := nn.linear_layer_new(4, 8), nn.linear_layer_new(8, 4)
		freeze(lin1)
		freeze(lin2)
		s3 := live()
		chain: [6]^t.Tensor
		for k in 0 ..< 3 {
			chain[2 * k] = nn.linear_forward(&lin1, x)
			chain[2 * k + 1] = nn.linear_forward(&lin2, chain[2 * k])
			step(&head, chain[2 * k + 1], targets)
		}
		left_bare := live() - s3
		for h in chain {t.tensor_free(h)} 	// what the unrecorded chain left
		s4 := live()
		for _ in 0 ..< 3 {
			rec := nn.forward_record_begin()
			h := nn.linear_forward(&lin2, nn.linear_forward(&lin1, x))
			loss := t.tensor_cross_entropy_loss(nn.linear_forward(&head, h), targets)
			nn.forward_record_end(rec, loss)
			t.tensor_backward(loss)
			t.tensor_free_graph(loss)
		}
		left_rec := live() - s4

		nn.ffn_layer_free(&ffn)
		nn.transformer_encoder_block_free(&enc)
		nn.mlp_free(&mlp)
		nn.linear_layer_free(&lin1)
		nn.linear_layer_free(&lin2)
		nn.linear_layer_free(&head)
		t.tensor_free(x)
		t.tensor_free(xs)
		report(
			fmt.tprintf(
				"frozen layers without a scope: FFN leaves only its outputs (%d in 3 steps; %d once freed by the caller), recorded encoder-block chain %d, MLP %d tensors; hand-written chain of 2 frozen linears: %d left without forward_record_*, %d with",
				ffn_out_only,
				left_ffn,
				left_enc,
				left_mlp,
				left_bare,
				left_rec,
			),
			ffn_out_only == 3 &&
			left_ffn == 0 &&
			left_enc == 0 &&
			left_mlp == 0 &&
			left_bare == 6 &&
			left_rec == 0,
			&passed,
			&total,
		)
		_ = allocs
	}

	{ 	// GRU = torch.nn.GRU with a nonzero bias_hh: its n part acts inside the
		// reset gate (bias_hn). Output, input / h_0 / parameter gradients; a
		// checkpoint keeps bias_hn; a GRU without it (older files) still loads
		gru := nn.gru_layer_new(3, 4)
		copy(gru.w_ih.data.data, NF_GRU_W_IH[:])
		copy(gru.w_hh.data.data, NF_GRU_W_HH[:])
		copy(gru.bias.data.data, NF_GRU_BIAS[:])
		copy(gru.bias_hn.data.data, NF_GRU_BIAS_HN[:])
		mk :: proc(v: []f64, shape: [4]int, grad: bool) -> ^t.Tensor {
			m := l.matrix_new(f64, 1, len(v))
			copy(m.data, v)
			x := t.tensor_new(m, grad)
			x.shape = shape
			return x
		}
		dev :: proc(a, b: []f64) -> f64 {
			if len(a) != len(b) {return math.inf_f64(1)}
			w := 0.0
			for v, i in a {w = max(w, abs(v - b[i]))}
			return w
		}
		x := mk(NF_GRU_X[:], {2, 5, 3, 1}, true)
		h0 := mk(NF_GRU_H0[:], {2, 4, 1, 1}, true)
		r := mk(NF_GRU_R[:], {2, 5, 4, 1}, false)
		out := nn.gru_layer_forward(&gru, x, h0)
		loss := t.tensor_sum(t.tensor_mul(out, r))
		t.tensor_backward(loss)
		d_out := dev(out.data.data, NF_GRU_OUT[:])
		d_grad := max(
			dev(x.grad.data, NF_GRU_DX[:]),
			dev(h0.grad.data, NF_GRU_DH0[:]),
			dev(gru.w_ih.grad.data, NF_GRU_DW_IH[:]),
			dev(gru.w_hh.grad.data, NF_GRU_DW_HH[:]),
			dev(gru.bias.grad.data, NF_GRU_DBIAS[:]),
			dev(gru.bias_hn.grad.data, NF_GRU_DBIAS_HN[:]),
		)
		t.tensor_free_graph(loss)
		report(
			fmt.tprintf(
				"GRU vs torch.nn.GRU with nonzero bias_hh and h_0: output %.1e, gradients (x, h_0, W_ih, W_hh, bias, bias_hn) %.1e",
				d_out,
				d_grad,
			),
			d_out < 1e-13 && d_grad < 1e-12,
			&passed,
			&total,
		)

		// checkpoint round trip: bias_hn is stored; then the same layer without
		// bias_hn (as in files written before it existed) saves and loads
		// without it and gives the same output as before
		path := "model_checkpoints/nn_fixes_gru.ckpt"
		model := nn.sequential_new()
		nn.sequential_add(model, gru)
		opt := nn.adam_new(0.01)
		nn.sequential_add_to_opt(model, &opt)
		saved := nn.save_checkpoint(model, &opt, path, 1)
		m2, o2, _, ok := nn.load_checkpoint(path)
		round := saved && ok
		if ok {
			g2 := m2.layers[0].(nn.GRULayer)
			round &&= g2.bias_hn != nil && slice.equal(g2.bias_hn.data.data, NF_GRU_BIAS_HN[:])
			round &&= len(nn.sequential_parameters(m2, context.temp_allocator)) == 4
			nn.sequential_free(m2)
			nn.adam_free(o2)
			free(o2)
		}
		g := &model.layers[0].(nn.GRULayer)
		t.tensor_free(g.bias_hn)
		g.bias_hn = nil
		x.requires_grad, h0.requires_grad = false, false
		ref := nn.gru_layer_forward(g, x, h0) // bias_hn nil = 0
		nn.adam_free(&opt)
		opt = nn.adam_new(0.01)
		nn.sequential_add_to_opt(model, &opt)
		saved = nn.save_checkpoint(model, &opt, path, 1)
		m3, o3, _, ok3 := nn.load_checkpoint(path)
		old := saved && ok3
		if ok3 {
			g3 := &m3.layers[0].(nn.GRULayer)
			old &&= g3.bias_hn == nil
			o := nn.gru_layer_forward(g3, x, h0)
			old &&= slice.equal(o.data.data, ref.data.data)
			old &&= len(o3.parameters) == 3
			t.tensor_free(o)
			nn.sequential_free(m3)
			nn.adam_free(o3)
			free(o3)
		}
		// sequential_load_partial puts it into a new GRU with bias_hn = 0
		fresh := nn.sequential_new()
		nn.sequential_add(fresh, nn.gru_layer_new(3, 4))
		fg := &fresh.layers[0].(nn.GRULayer)
		for &v in fg.bias_hn.data.data {v = 1}
		n_loaded, n_skipped, ok_p := nn.sequential_load_partial(fresh, path)
		old &&= ok_p && n_loaded == 1 && n_skipped == 0
		old &&=
			slice.equal(fg.w_hh.data.data, NF_GRU_W_HH[:]) &&
			slice.equal(fg.bias.data.data, NF_GRU_BIAS[:])
		for v in fg.bias_hn.data.data {old &&= v == 0}
		nn.sequential_free(fresh)
		t.tensor_free(ref)
		_ = os.remove(path)
		report(
			fmt.tprintf(
				"GRU checkpoint: bias_hn saved and loaded %v; a GRU without bias_hn saves and loads without it (same output, 3 parameters) and loads partially into a new GRU with bias_hn = 0 %v",
				round,
				old,
			),
			round && old,
			&passed,
			&total,
		)
		nn.adam_free(&opt)
		nn.sequential_free(model)
		t.tensor_free(x)
		t.tensor_free(h0)
		t.tensor_free(r)
	}

	{ 	// tanh-approximated GELU (GPT-2's gelu_new) = torch gelu(approximate="tanh"),
		// value and gradient; FFN / GPT keep the choice in their files, and files
		// without it (older FFN layer and GPT ids) load with exact GELU
		xm := l.matrix_new(f64, 1, len(NF_GT_X))
		copy(xm.data, NF_GT_X[:])
		x := t.tensor_new(xm, true)
		wm := l.matrix_new(f64, 1, len(NF_GT_W))
		copy(wm.data, NF_GT_W[:])
		w := t.tensor_new(wm, false)
		y := t.tensor_gelu_tanh(x)
		loss := t.tensor_sum(t.tensor_mul(y, w))
		t.tensor_backward(loss)
		d_y, d_dx := 0.0, 0.0
		for v, i in y.data.data {d_y = max(d_y, abs(v - NF_GT_Y[i]) / max(1, abs(NF_GT_Y[i])))}
		for v, i in x.grad.data {d_dx = max(d_dx, abs(v - NF_GT_DX[i]) / max(1, abs(NF_GT_DX[i])))}
		t.tensor_free_graph(loss)
		t.tensor_free(x)
		t.tensor_free(w)
		report(
			fmt.tprintf(
				"tensor_gelu_tanh = torch gelu(approximate=tanh), x in [-6, 12]: value %.1e, gradient %.1e",
				d_y,
				d_dx,
			),
			d_y < 1e-15 && d_dx < 1e-15,
			&passed,
			&total,
		)

		// Sequential checkpoint: FFN(gelu_tanh) and Activation.GeluTanh
		path := "model_checkpoints/nn_fixes_gelu.ckpt"
		model := nn.sequential_new()
		nn.sequential_add(
			model,
			nn.ffn_layer_new(4, 8, gelu_tanh = true),
			nn.Activation.GeluTanh,
			nn.linear_layer_new(4, 2),
		)
		opt := nn.adam_new(0.01)
		nn.sequential_add_to_opt(model, &opt)
		seq_ok := nn.save_checkpoint(model, &opt, path, 1)
		out1 := nn.sequential_forward(model, nf_x)
		m2, o2, _, ok2 := nn.load_checkpoint(path)
		seq_ok &&= ok2
		if ok2 {
			seq_ok &&=
				m2.layers[0].(nn.FFNLayer).gelu_tanh && m2.layers[1].(nn.Activation) == .GeluTanh
			out2 := nn.sequential_forward(m2, nf_x)
			seq_ok &&= slice.equal(out1.data.data, out2.data.data)
			t.tensor_free_graph(out2)
			nn.sequential_free(m2)
			nn.adam_free(o2)
			free(o2)
		}
		t.tensor_free_graph(out1)
		// the same file with the older FFN layer id (14, no flag) loads with exact GELU
		data, _ := os.read_entire_file(path, context.allocator)
		old_ffn := slice.concatenate([][]u8{data[:18], {14, 0, 0, 0}, data[22:30], data[34:]})
		_ = os.write_entire_file(path, old_ffn)
		m3, o3, _, ok3 := nn.load_checkpoint(path)
		ffn0_ok := ok3
		if ok3 {
			ffn0_ok &&= !m3.layers[0].(nn.FFNLayer).gelu_tanh
			nn.sequential_free(m3)
			nn.adam_free(o3)
			free(o3)
		}
		delete(old_ffn)
		delete(data)
		nn.adam_free(&opt)
		nn.sequential_free(model)

		// GPT file: gelu_tanh is stored; a file of the previous GPT id (33, no
		// GELU flag) loads with exact GELU
		gpath := "model_checkpoints/nn_fixes_gelu.gpt"
		gpt := nn.gpt_model_new(11, 8, 2, 16, 2, 6, gelu_tanh = true)
		gpt_ok := nn.save_gpt_model(&gpt, gpath)
		g2, okg := nn.load_gpt_model(gpath)
		gpt_ok &&= okg
		if okg {
			gpt_ok &&= g2.gelu_tanh && g2.blocks[0].ffn.gelu_tanh
			mask := nn.create_causal_mask(6, context.allocator)
			a := nn.gpt_model_forward(&gpt, nf_ids_t, mask, false)
			b := nn.gpt_model_forward(g2, nf_ids_t, mask, false)
			gpt_ok &&= slice.equal(a.data.data, b.data.data)
			t.tensor_free_graph(a)
			t.tensor_free_graph(b)
			delete(mask)
			nn.gpt_model_free(g2)
			free(g2)
		}
		gdata, _ := os.read_entire_file(gpath, context.allocator)
		old_gpt := slice.concatenate([][]u8{gdata[:10], {33, 0, 0, 0}, gdata[14:54], gdata[58:]})
		_ = os.write_entire_file(gpath, old_gpt)
		g3, okg3 := nn.load_gpt_model(gpath)
		gpt3_ok := okg3
		if okg3 {
			gpt3_ok &&= !g3.gelu_tanh && !g3.blocks[0].ffn.gelu_tanh
			nn.gpt_model_free(g3)
			free(g3)
		}
		delete(old_gpt)
		delete(gdata)
		nn.gpt_model_free(&gpt)
		_ = os.remove(path)
		_ = os.remove(gpath)
		report(
			fmt.tprintf(
				"tanh GELU kept by checkpoints (FFN + Activation %v, GPT %v); older FFN / GPT files load with exact GELU %v %v",
				seq_ok,
				gpt_ok,
				ffn0_ok,
				gpt3_ok,
			),
			seq_ok && gpt_ok && ffn0_ok && gpt3_ok,
			&passed,
			&total,
		)
	}

	{ 	// reparameterize requires grad only if mu or log_var does
		mk :: proc(grad: bool) -> ^t.Tensor {
			m := l.matrix_new(f64, 2, 3)
			for &v, i in m.data {v = 0.1 * f64(i) - 0.2}
			return t.tensor_new(m, grad)
		}
		mu0, lv0 := mk(false), mk(false)
		z0 := nn.reparameterize(mu0, lv0, context.allocator)
		const_ok := !z0.requires_grad && z0.op == .None && len(z0.inputs) == 0
		t.tensor_free(z0)
		mu1 := mk(true)
		z1 := nn.reparameterize(mu1, lv0, context.allocator)
		loss := t.tensor_sum(z1)
		t.tensor_backward(loss)
		grad_ok := z1.requires_grad && z1.op == .Reparameterize && mu1.grad.data[0] == 1
		t.tensor_free_graph(loss)
		for x in ([]^t.Tensor{mu0, lv0, mu1}) {t.tensor_free(x)}
		report(
			fmt.tprintf(
				"reparameterize: constant from inputs without grad %v, op with gradient to mu %v",
				const_ok,
				grad_ok,
			),
			const_ok && grad_ok,
			&passed,
			&total,
		)
	}

	t.tensor_free(nf_ids_t)
	t.tensor_free(nf_seg_t)
	t.tensor_free(nf_x)
	report(
		fmt.tprintf("all models freed: %d live allocations", len(track.allocation_map)),
		len(track.allocation_map) == 0,
		&passed,
		&total,
	)

	fmt.printf("nn regression tests: %d/%d passed\n", passed, total)
	record_failures(total - passed)
}
