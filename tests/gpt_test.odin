package tests

import l "../wotan/linalg"
import nn "../wotan/nn"
import t "../wotan/tensor"
import "base:runtime"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

import "core:mem/virtual"
gpt_full_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== Full GPT Implementation Test ===")

	// Base training text
	base_text := "to be or not to be that is the question whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune or to take arms against a sea of troubles and by opposing end them to die to sleep no more and by a sleep to say we end the heartache and the thousand natural shocks that flesh is heir to tis a consummation devoutly to be wished to die to sleep to sleep perchance to dream ay theres the rub for in that sleep of death what dreams may come when we have shuffled off this mortal coil must give us pause theres the respect that makes calamity of so long life for who would bear the whips and scorns of time the oppressors wrong the proud mans contumely the pangs of despised love the laws delay the insolence of office and the spurns that patient merit of the unworthy takes when he himself might his quietus make with a bare bodkin who would fardels bear to grunt and sweat under a weary life but that the dread of something after death the undiscovered country from whose bourn no traveller returns puzzles the will and makes us rather bear those ills we have than fly to others that we know not of thus conscience does make cowards of us all and thus the native hue of resolution is sicklied oer with the pale cast of thought and enterprises of great pith and moment with this regard their currents turn awry and lose the name of action soft you now the fair ophelia nymph in thy orisons be all my sins remembered my good lord how does your honour for this long time my lord i have had no answer would you vouchsafe to speak with me my lord i have a suit to you my lord the queen would speak with you and presently my lord i shall obey the queen my lord the king your father and the queen your mother are in a most sad plight the king my lord the queen your mother in most great addition of sorrow comes to your highness the queen your mother would speak with you presently the queen my lord the king your father and the queen your mother are in a most sad plight the king my lord the queen your mother in most great addition of sorrow comes to your highness"

	// ✅ Build augmented text using dynamic byte array
	text := make([dynamic]u8, 0, allocator)
	defer delete(text)

	// Repeat the base text 20 times with spaces
	for i in 0 ..< 20 {
		for c in base_text {
			append(&text, u8(c))
		}
		if i < 19 {
			append(&text, u8(' '))
		}
	}

	// Build vocabulary
	vocab := make(map[u8]int, allocator)
	inv_vocab := make(map[int]u8, allocator)
	vocab_size := 0

	for c in text {
		if !(c in vocab) {
			vocab[c] = vocab_size
			inv_vocab[vocab_size] = c
			vocab_size += 1
		}
	}

	fmt.printf("Vocabulary size: %d characters\n", vocab_size)
	fmt.printf("Text length: %d characters\n", len(text))

	// Model hyperparameters
	seq_len := 32
	d_model := 64
	num_heads := 4
	d_ff := 256
	num_layers := 2
	batch_size := 16
	max_seq_len := seq_len + 1
	// After building vocabulary, before model creation:
	fmt.println("\n=== Creating Model ===")
	global_tensors_created := 0

	// Create GPT model
	model := nn.gpt_model_new(
		vocab_size,
		d_model,
		num_heads,
		d_ff,
		num_layers,
		max_seq_len,
		allocator,
	)
	fmt.printf("Model created %d tensors\n", global_tensors_created)
	defer nn.gpt_model_free(&model)

	// Create optimizer
	fmt.println("\n=== Creating Optimizer ===")
	global_tensors_created = 0
	opt := nn.adam_new(0.0003, allocator = allocator)
	nn.gpt_model_add_to_optimizer(&model, &opt)
	fmt.printf("Optimizer setup created %d tensors\n", global_tensors_created)
	defer nn.adam_free(&opt)

	// Create causal mask
	fmt.println("\n=== Creating Causal Mask ===")
	global_tensors_created = 0
	mask := nn.create_causal_mask(max_seq_len, allocator)
	fmt.printf("Causal mask created %d tensors\n", global_tensors_created)
	defer delete(mask, allocator)

	// Convert text to indices
	fmt.println("\n=== Converting Text to Indices ===")
	global_tensors_created = 0
	text_indices := make([]int, len(text), allocator)
	for i in 0 ..< len(text) {
		text_indices[i] = vocab[text[i]]
	}
	fmt.printf("Text conversion created %d tensors\n", global_tensors_created)
	defer delete(text_indices, allocator)

	fmt.println("\n=== Starting Training ===")

	// Training loop
	epochs := 1500
	fmt.printf(
		"Training GPT (layers=%d, d_model=%d, heads=%d)...\n",
		num_layers,
		d_model,
		num_heads,
	)

	initial_loss := 0.0
	final_loss := 0.0
	arena: virtual.Arena
	err := virtual.arena_init_static(&arena, 256 * mem.Megabyte)
	if err != nil {
		fmt.printf("Failed to initialize arena: %v\n", err)
		return
	}
	defer virtual.arena_destroy(&arena)
	arena_alloc := virtual.arena_allocator(&arena)
	// In your training loop, add validation:
	for epoch in 0 ..< epochs {
		nn.adam_zero_grad(&opt)

		// Sample random sequences from text
		input_ids_data := l.matrix_new(f64, 1, batch_size * seq_len, allocator)
		target_indices := make([]int, batch_size * seq_len, allocator)

		for b in 0 ..< batch_size {
			start := int(rand.int31()) % (len(text) - seq_len - 1)

			for s in 0 ..< seq_len {
				input_ids_data.data[b * seq_len + s] = f64(text_indices[start + s])
				target_indices[b * seq_len + s] = text_indices[start + s + 1]
			}
		}

		input_ids := t.tensor_new(input_ids_data, false, allocator)
		input_ids.shape = [4]int{batch_size, seq_len, 1, 1}

		// ✅ ADD: Validate input
		t.tensor_validate(input_ids, "input_ids before forward")

		// Forward pass with training=true for dropout
		logits := nn.gpt_model_forward(&model, input_ids, mask, true)

		// ✅ ADD: Validate logits
		t.tensor_validate(logits, "logits after forward")

		// Compute cross-entropy loss
		loss := t.tensor_cross_entropy_loss(logits, target_indices)

		// ✅ ADD: Validate loss
		t.tensor_validate(loss, "loss after computation")

		if epoch == 0 {
			initial_loss = loss.data.data[0]
		}

		// Backward pass
		t.tensor_backward(loss, arena_alloc)
		virtual.arena_free_all(&arena)
		// ✅ ADD: Gradient clipping to prevent explosion
		for param in opt.parameters {
			if param.grad.data != nil {
				grad_norm := 0.0
				for val in param.grad.data {
					grad_norm += val * val
				}
				grad_norm = math.sqrt(grad_norm)

				max_grad_norm := 1.0
				if grad_norm > max_grad_norm {
					scale := max_grad_norm / grad_norm
					for i in 0 ..< len(param.grad.data) {
						param.grad.data[i] *= scale
					}
				}
			}
		}

		// ✅ ADD: Check for NaN/Inf in loss
		if math.is_nan(loss.data.data[0]) || math.is_inf(loss.data.data[0]) {
			fmt.printf("WARNING: Loss is NaN/Inf at epoch %d, stopping training\n", epoch)
			break
		}
		base_lr := 0.0003
		if epoch < 100 {
			// Warmup phase
			opt.learning_rate = base_lr * f64(epoch + 1) / 100.0
		} else {
			// Cosine decay
			progress := f64(epoch - 100) / f64(epochs - 100)
			opt.learning_rate = base_lr * 0.5 * (1.0 + math.cos(math.PI * progress))
		}
		// Optimizer step
		nn.adam_step(&opt)

		if epoch % 60 == 0 {
			perplexity := math.exp(loss.data.data[0])
			fmt.printf(
				"Epoch %d | Loss: %.4f | Perplexity: %.2f\n",
				epoch,
				loss.data.data[0],
				perplexity,
			)
		}

		if epoch == epochs - 1 {
			final_loss = loss.data.data[0]
		}

		// Clean up
		t.tensor_free_graph(loss)
		t.tensor_free(input_ids)
		delete(target_indices, allocator)

		// ✅ ADD: Periodic validation of model parameters
		// In your training loop, replace the validation section with this:
		if epoch % 100 == 0 {
			// Check a few key parameters without creating strings
			for i in 0 ..< len(model.blocks) {
				block := &model.blocks[i]

				// Direct validation without string formatting
				if block.mha.q_proj.weights == nil ||
				   len(block.mha.q_proj.weights.data.data) == 0 {
					fmt.printf("ERROR: block[%d].mha.q_proj invalid at epoch %d\n", i, epoch)
				}
				if block.mha.k_proj.weights == nil ||
				   len(block.mha.k_proj.weights.data.data) == 0 {
					fmt.printf("ERROR: block[%d].mha.k_proj invalid at epoch %d\n", i, epoch)
				}
				if block.mha.v_proj.weights == nil ||
				   len(block.mha.v_proj.weights.data.data) == 0 {
					fmt.printf("ERROR: block[%d].mha.v_proj invalid at epoch %d\n", i, epoch)
				}
			}
		}
	}

	reduction := (1.0 - final_loss / initial_loss) * 100.0
	final_perplexity := math.exp(final_loss)
	fmt.printf(
		"\n✓ Training complete! Loss: %.4f → %.4f (%.1f%% reduction)\n",
		initial_loss,
		final_loss,
		reduction,
	)
	fmt.printf("Final perplexity: %.2f\n", final_perplexity)

	// Generate text with different sampling strategies
	fmt.println("\n=== Generating Text ===")

	SamplingConfig :: struct {
		name:   string,
		method: string,
		param:  f64,
	}

	sampling_configs := []SamplingConfig {
		{"Temperature (0.7)", "temperature", 0.7},
		{"Temperature (1.0)", "temperature", 1.0},
		{"Top-k (k=20)", "top_k", 20.0},
		{"Top-p (p=0.85)", "top_p", 0.85},
	}

	for config in sampling_configs {
		fmt.printf("\n%s:\n", config.name)

		// Start with "to be"
		seed := "to be"
		generated := make([dynamic]u8, 0, allocator)

		for c in seed {
			append(&generated, u8(c))
		}

		// Generate 200 characters
		// Generate 200 characters
		// Generate 200 characters
		recent_tokens := make([dynamic]int, 0, allocator)
		defer delete(recent_tokens)

		for i in 0 ..< 200 {
			// Prepare input sequence
			gen_input := l.matrix_new(f64, 1, seq_len, allocator)

			gen_len := len(generated)
			if gen_len > seq_len {
				gen_len = seq_len
			}

			for s in 0 ..< gen_len {
				offset := len(generated) - gen_len + s
				gen_input.data[s] = f64(vocab[generated[offset]])
			}

			gen_ids := t.tensor_new(gen_input, false, allocator)
			gen_ids.shape = [4]int{1, seq_len, 1, 1}

			// Forward pass with training=false (no dropout)
			gen_logits := nn.gpt_model_forward(&model, gen_ids, mask, false)

			// Get logits for last position
			last_logits := make([]f64, vocab_size, allocator)
			for v in 0 ..< vocab_size {
				last_logits[v] = gen_logits.data.data[(seq_len - 1) * vocab_size + v]
			}

			// Sample based on method with repetition penalty
			// Sample based on method with repetition penalty
			next_char_idx := 0
			if config.method == "temperature" {
				next_char_idx = nn.gpt_sample_temperature(
					last_logits,
					config.param,
					recent_tokens[:], // ✅ Pass as slice
					1.2,
				)
			} else if config.method == "top_k" {
				next_char_idx = nn.gpt_sample_top_k(
					last_logits,
					int(config.param),
					1.0,
					recent_tokens[:], // ✅ Pass as slice
					1.2,
				)
			} else if config.method == "top_p" {
				next_char_idx = nn.gpt_sample_top_p(
					last_logits,
					config.param,
					1.0,
					recent_tokens[:], // ✅ Pass as slice
					1.2,
				)
			}

			append(&generated, inv_vocab[next_char_idx])

			// Track recent tokens (keep last 10)
			// Track recent tokens (keep last 10)
			append(&recent_tokens, next_char_idx)

			// Remove oldest by shifting and resizing
			if len(recent_tokens) > 10 {
				// Shift all elements left by 1
				for j in 0 ..< len(recent_tokens) - 1 {
					recent_tokens[j] = recent_tokens[j + 1]
				}
				// Properly remove last element from dynamic array
				new_tokens := make([dynamic]int, len(recent_tokens) - 1, allocator)
				for j in 0 ..< len(recent_tokens) - 1 {
					new_tokens[j] = recent_tokens[j]
				}
				delete(recent_tokens)
				recent_tokens = new_tokens
			}

			// Clean up
			l.matrix_free(&gen_input)
			delete(last_logits, allocator)
			t.tensor_free(gen_ids)
			t.tensor_free(gen_logits)
		}

		fmt.printf("%.*s\n", len(generated), generated[:])
		delete(generated)
	}

	fmt.println("\n✓ Full GPT test completed!")
}


gpt_shakespeare_test :: proc(allocator: mem.Allocator) {
	fmt.println("\n=== GPT Shakespeare Corpus Test ===")

	// Large Shakespeare corpus - Hamlet and other famous passages
	shakespeare_text := `
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life.
For who would bear the whips and scorns of time,
Th'oppressor's wrong, the proud man's contumely,
The pangs of despised love, the law's delay,
The insolence of office, and the spurns
That patient merit of th'unworthy takes,
When he himself might his quietus make
With a bare bodkin? Who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscover'd country from whose bourn
No traveller returns, puzzles the will,
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all,
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pith and moment
With this regard their currents turn awry
And lose the name of action.

Now is the winter of our discontent
Made glorious summer by this sun of York;
And all the clouds that lour'd upon our house
In the deep bosom of the ocean buried.
Now are our brows bound with victorious wreaths;
Our bruised arms hung up for monuments;
Our stern alarums changed to merry meetings,
Our dreadful marches to delightful measures.
Grim-visaged war hath smooth'd his wrinkled front;
And now, instead of mounting barded steeds
To fright the souls of fearful adversaries,
He capers nimbly in a lady's chamber
To the lascivious pleasing of a lute.

But I, that am not shaped for sportive tricks,
Nor made to court an amorous looking-glass;
I, that am rudely stamp'd, and want love's majesty
To strut before a wanton ambling nymph;
I, that am curtail'd of this fair proportion,
Cheated of feature by dissembling nature,
Deformed, unfinish'd, sent before my time
Into this breathing world, scarce half made up,
And that so lamely and unfashionable
That dogs bark at me as I halt by them;
Why, I, in this weak piping time of peace,
Have no delight to pass away the time,
Unless to spy my shadow in the sun
And descant on mine own deformity:
And therefore, since I cannot prove a lover,
To entertain these fair well-spoken days,
I am determined to prove a villain
And hate the idle pleasures of these days.

Friends, Romans, countrymen, lend me your ears;
I come to bury Caesar, not to praise him.
The evil that men do lives after them;
The good is oft interred with their bones;
So let it be with Caesar. The noble Brutus
Hath told you Caesar was ambitious:
If it were so, it was a grievous fault,
And grievously hath Caesar answer'd it.
Here, under leave of Brutus and the rest
For Brutus is an honourable man;
So are they all, all honourable men,
Come I to speak in Caesar's funeral.
He was my friend, faithful and just to me:
But Brutus says he was ambitious;
And Brutus is an honourable man.

O Romeo, Romeo, wherefore art thou Romeo?
Deny thy father and refuse thy name;
Or, if thou wilt not, be but sworn my love,
And I'll no longer be a Capulet.
'Tis but thy name that is my enemy;
Thou art thyself, though not a Montague.
What's Montague? it is nor hand, nor foot,
Nor arm, nor face, nor any other part
Belonging to a man. O, be some other name!
What's in a name? that which we call a rose
By any other name would smell as sweet;
So Romeo would, were he not Romeo call'd,
Retain that dear perfection which he owes
Without that title. Romeo, doff thy name,
And for that name which is no part of thee
Take all myself.

All the world's a stage,
And all the men and women merely players;
They have their exits and their entrances,
And one man in his time plays many parts,
His acts being seven ages. At first, the infant,
Mewling and puking in the nurse's arms.
Then the whining schoolboy, with his satchel
And shining morning face, creeping like snail
Unwillingly to school. And then the lover,
Sighing like furnace, with a woeful ballad
Made to his mistress' eyebrow. Then a soldier,
Full of strange oaths and bearded like the pard,
Jealous in honor, sudden and quick in quarrel,
Seeking the bubble reputation
Even in the cannon's mouth. And then the justice,
In fair round belly with good capon lined,
With eyes severe and beard of formal cut,
Full of wise saws and modern instances;
And so he plays his part. The sixth age shifts
Into the lean and slippered pantaloon,
With spectacles on nose and pouch on side;
His youthful hose, well saved, a world too wide
For his shrunk shank, and his big manly voice,
Turning again toward childish treble, pipes
And whistles in his sound. Last scene of all,
That ends this strange eventful history,
Is second childishness and mere oblivion,
Sans teeth, sans eyes, sans taste, sans everything.

If music be the food of love, play on,
Give me excess of it; that surfeiting,
The appetite may sicken, and so die.
That strain again, it had a dying fall;
O, it came o'er my ear like the sweet sound
That breathes upon a bank of violets,
Stealing and giving odour. Enough, no more,
'Tis not so sweet now as it was before.
O spirit of love, how quick and fresh art thou,
That, notwithstanding thy capacity
Receiveth as the sea, nought enters there,
Of what validity and pitch soe'er,
But falls into abatement and low price
Even in a minute. So full of shapes is fancy
That it alone is high fantastical.

The quality of mercy is not strained.
It droppeth as the gentle rain from heaven
Upon the place beneath. It is twice blest;
It blesseth him that gives and him that takes.
'Tis mightiest in the mightiest; it becomes
The throned monarch better than his crown.
His sceptre shows the force of temporal power,
The attribute to awe and majesty,
Wherein doth sit the dread and fear of kings;
But mercy is above this sceptred sway;
It is enthroned in the hearts of kings,
It is an attribute to God himself;
And earthly power doth then show likest God's
When mercy seasons justice. Therefore, Jew,
Though justice be thy plea, consider this,
That, in the course of justice, none of us
Should see salvation. We do pray for mercy,
And that same prayer doth teach us all to render
The deeds of mercy.

Shall I compare thee to a summer's day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date:
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimm'd;
And every fair from fair sometime declines,
By chance or nature's changing course untrimm'd;
But thy eternal summer shall not fade
Nor lose possession of that fair thou owest;
Nor shall Death brag thou wander'st in his shade,
When in eternal lines to time thou growest.
So long as men can breathe or eyes can see,
So long lives this and this gives life to thee.

We are such stuff
As dreams are made on, and our little life
Is rounded with a sleep.

The lady doth protest too much, methinks.

Though this be madness, yet there is method in't.

What's in a name? That which we call a rose
By any other name would smell as sweet.

Good night, good night! Parting is such sweet sorrow,
That I shall say good night till it be morrow.

Out, damned spot! out, I say!

By the pricking of my thumbs,
Something wicked this way comes.

The course of true love never did run smooth.

Lord, what fools these mortals be!

Double, double toil and trouble;
Fire burn and caldron bubble.

Cry 'Havoc!' and let slip the dogs of war.

Et tu, Brute?

Brevity is the soul of wit.

To thine own self be true.

This above all: to thine own self be true,
And it must follow, as the night the day,
Thou canst not then be false to any man.

Neither a borrower nor a lender be;
For loan oft loses both itself and friend,
And borrowing dulls the edge of husbandry.
`

	// Build vocabulary
	vocab := make(map[u8]int, allocator)
	inv_vocab := make(map[int]u8, allocator)
	vocab_size := 0

	for c in shakespeare_text {
		c_u8 := u8(c)
		if !(c_u8 in vocab) {
			vocab[c_u8] = vocab_size
			inv_vocab[vocab_size] = c_u8
			vocab_size += 1
		}
	}

	fmt.printf("Vocabulary size: %d characters\n", vocab_size)
	fmt.printf("Text length: %d characters\n", len(shakespeare_text))

	// Model hyperparameters - same as working test
	seq_len := 32
	d_model := 64
	num_heads := 4
	d_ff := 256
	num_layers := 2
	batch_size := 16
	max_seq_len := seq_len + 1

	// Create GPT model
	model := nn.gpt_model_new(
		vocab_size,
		d_model,
		num_heads,
		d_ff,
		num_layers,
		max_seq_len,
		allocator,
	)
	defer nn.gpt_model_free(&model)

	// Create optimizer
	opt := nn.adam_new(0.0003, allocator = allocator)
	defer nn.adam_free(&opt)

	// Register all parameters
	nn.gpt_model_add_to_optimizer(&model, &opt)

	// Create causal mask
	mask := nn.create_causal_mask(max_seq_len, allocator)
	defer delete(mask, allocator)

	// Convert text to indices
	text_indices := make([]int, len(shakespeare_text), allocator)
	defer delete(text_indices, allocator)

	for i in 0 ..< len(shakespeare_text) {
		text_indices[i] = vocab[u8(shakespeare_text[i])]
	}

	// Training loop - 1500 epochs (stable limit)
	epochs := 1500
	fmt.printf(
		"Training GPT on Shakespeare (layers=%d, d_model=%d, heads=%d)...\n",
		num_layers,
		d_model,
		num_heads,
	)

	initial_loss := 0.0
	final_loss := 0.0
	arena: virtual.Arena
	err := virtual.arena_init_static(&arena, 256 * mem.Megabyte)
	if err != nil {
		fmt.printf("Failed to initialize arena: %v\n", err)
		return
	}
	defer virtual.arena_destroy(&arena)

	arena_alloc := virtual.arena_allocator(&arena)

	for epoch in 0 ..< epochs {
		nn.adam_zero_grad(&opt)

		// Sample random sequences from text
		input_ids_data := l.matrix_new(f64, 1, batch_size * seq_len, allocator)
		target_indices := make([]int, batch_size * seq_len, allocator)

		for b in 0 ..< batch_size {
			start := int(rand.int31()) % (len(shakespeare_text) - seq_len - 1)

			for s in 0 ..< seq_len {
				input_ids_data.data[b * seq_len + s] = f64(text_indices[start + s])
				target_indices[b * seq_len + s] = text_indices[start + s + 1]
			}
		}

		input_ids := t.tensor_new(input_ids_data, false, allocator)
		input_ids.shape = [4]int{batch_size, seq_len, 1, 1}

		// Forward pass with training=true for dropout
		logits := nn.gpt_model_forward(&model, input_ids, mask, true)

		// Compute cross-entropy loss
		loss := t.tensor_cross_entropy_loss(logits, target_indices)

		if epoch == 0 {
			initial_loss = loss.data.data[0]
		}

		// Backward pass
		t.tensor_backward(loss, arena_alloc)
		virtual.arena_free_all(&arena)
		// Optimizer step
		nn.adam_step(&opt)

		if epoch % 100 == 0 {
			perplexity := math.exp(loss.data.data[0])
			fmt.printf(
				"Epoch %d | Loss: %.4f | Perplexity: %.2f\n",
				epoch,
				loss.data.data[0],
				perplexity,
			)
		}

		if epoch == epochs - 1 {
			final_loss = loss.data.data[0]
		}

		// Clean up
		t.tensor_free_graph(loss)
		t.tensor_free(input_ids)
		delete(target_indices, allocator)
	}

	reduction := (1.0 - final_loss / initial_loss) * 100.0
	final_perplexity := math.exp(final_loss)
	fmt.printf(
		"\n✓ Training complete! Loss: %.4f → %.4f (%.1f%% reduction)\n",
		initial_loss,
		final_loss,
		reduction,
	)
	fmt.printf("Final perplexity: %.2f\n", final_perplexity)

	// Generate text with different sampling strategies
	fmt.println("\n=== Generating Shakespeare ===")

	SamplingConfig :: struct {
		name:   string,
		method: string,
		param:  f64,
	}

	sampling_configs := []SamplingConfig {
		{"Temperature (0.7)", "temperature", 0.7},
		{"Temperature (1.0)", "temperature", 1.0},
		{"Top-k (k=20)", "top_k", 20.0},
		{"Top-p (p=0.9)", "top_p", 0.9},
	}

	for config in sampling_configs {
		fmt.printf("\n%s:\n", config.name)

		// Start with "To be"
		seed := "To be"
		generated := make([dynamic]u8, 0, allocator)

		for c in seed {
			append(&generated, u8(c))
		}

		// Generate 300 characters (longer for more diverse text)
		recent_tokens := make([dynamic]int, 0, allocator)

		for i in 0 ..< 300 {
			// Prepare input sequence
			gen_input := l.matrix_new(f64, 1, seq_len, allocator)

			gen_len := len(generated)
			if gen_len > seq_len {
				gen_len = seq_len
			}

			for s in 0 ..< gen_len {
				offset := len(generated) - gen_len + s
				gen_input.data[s] = f64(vocab[generated[offset]])
			}

			gen_ids := t.tensor_new(gen_input, false, allocator)
			gen_ids.shape = [4]int{1, seq_len, 1, 1}

			// Forward pass with training=false (no dropout)
			gen_logits := nn.gpt_model_forward(&model, gen_ids, mask, false)

			// Get logits for last position
			last_logits := make([]f64, vocab_size, allocator)
			for v in 0 ..< vocab_size {
				last_logits[v] = gen_logits.data.data[(seq_len - 1) * vocab_size + v]
			}

			// Sample based on method with repetition penalty
			next_char_idx := 0
			if config.method == "temperature" {
				next_char_idx = nn.gpt_sample_temperature(
					last_logits,
					config.param,
					recent_tokens[:],
					1.2,
				)
			} else if config.method == "top_k" {
				next_char_idx = nn.gpt_sample_top_k(
					last_logits,
					int(config.param),
					1.0,
					recent_tokens[:],
					1.2,
				)
			} else if config.method == "top_p" {
				next_char_idx = nn.gpt_sample_top_p(
					last_logits,
					config.param,
					1.0,
					recent_tokens[:],
					1.2,
				)
			}

			append(&generated, inv_vocab[next_char_idx])

			// Track recent tokens (keep last 10)
			append(&recent_tokens, next_char_idx)

			if len(recent_tokens) > 10 {
				new_tokens := make([dynamic]int, len(recent_tokens) - 1, allocator)
				for j in 0 ..< len(recent_tokens) - 1 {
					new_tokens[j] = recent_tokens[j + 1]
				}
				delete(recent_tokens)
				recent_tokens = new_tokens
			}

			// Clean up
			l.matrix_free(&gen_input)
			delete(last_logits, allocator)
			t.tensor_free(gen_ids)
			t.tensor_free(gen_logits)
		}

		fmt.printf("%.*s\n", len(generated), generated[:])
		delete(generated)
		delete(recent_tokens)
	}

	fmt.println("\n✓ Shakespeare GPT test completed!")
}
