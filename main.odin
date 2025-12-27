// main.odin
// 1.58 Bit - style fully connected MNIST trainer / inferencer ( IDX ubyte files ),
// parallelized ( with pthreads ).
//
// Build:
//   odin build . -o:speed -out:bitnet_mnist.exe -no-bounds-check
//
// Train:
//   ./bitnet_mnist.exe --epochs 3 --batch 64 --hidden 256 --lr 0.001 --threads 24
//
// train ( batch 512 is a good traide off for high performance):
//   time ./bitnet_mnist.exe \
//   --train-images mnist_idx/train-images-idx3-ubyte \
//   --train-labels mnist_idx/train-labels-idx1-ubyte \
//   --test-images  mnist_idx/t10k-images-idx3-ubyte \
//   --test-labels  mnist_idx/t10k-labels-idx1-ubyte \
//   --epochs 30 --batch 512 --hidden 512 \
//   --lr 0.0005 --lr_start 0.005 --lr_stop 0.0002 \
//   --threads 24
//
// Inference:
//   ./bitnet_mnist.exe --weights weights.bla --infer-index 123
//
// Notes:
// - This uses core:sys/posix pthreads ( Linux ).
// - Quantized forward ( ternary weights + int8 activations ).
// - Backward uses STE ( FP32 linear approximation ).
//
// License:
//     MIT Open Source License
//

package main

import "core:fmt"
import "core:math"
import "core:os"
import "core:time"
import "core:sys/posix"
import "base:runtime"
import "core:strconv"

// BLA_EPS :: 1e-5
BLA_EPS :: 1e-6

QN      :: -128
QP      ::  127

BLA_MAGIC :: u32( 0x31414C42 ) // 'B''L''A''1' little-endian

die :: proc( msg : string ) {

	fmt.eprintln( msg )
	os.exit( -1 )
}

die2 :: proc ( msg : string,
               a   : string ) {

	fmt.println( msg, a )
	os.exit( 1 )
}

clampi :: proc "contextless" ( x  : int,
                               lo : int,
                               hi : int ) ->
                               int {

    if x < lo {

        return lo
    }
	if x > hi {

	    return hi
	}
	return x
}

zeros_f32 :: proc "contextless" ( x : [ ]f32 ) {

	for i in 0 ..< len( x ) {

	    x[ i ] = 0
	}
}

zeros_i32 :: proc "contextless" ( x : [ ]i32 ) {

	for i in 0 ..< len( x ) {

	    x[ i ] = 0
	}
}

// Round-to-nearest, ties-to-even (approx lrintf under default FE_TONEAREST)
lrintf_even :: proc "contextless" ( x : f32 ) ->
                                    int {

    xf := f64( x )

	if xf >= 0 {

		i := int( math.floor( xf ) )
		f := xf - f64( i )
		if f > 0.5  do return i + 1
		if f < 0.5  do return i
		// tie
		if ( i & 1 ) != 0 do return i + 1
		return i

	} else {

		// For negative, floor goes to more negative; handle with symmetry
		ax := -xf
		i  := int( math.floor( ax ) )
		f  := ax - f64( i )
		r  := 0
		if f > 0.5  {

		    r = i + 1
	    } else if f < 0.5 {

		    r = i
	    } else {

			if ( i & 1 ) != 0 {

			    r = i + 1
			} else {

			    r = i
		    }
		}

		return -r
	}
}

// Simple RNG (xorshift64*) + Box-Muller

rng_state : u64 = 88172645463393265

rng_seed :: proc "contextless" ( seed: u64) {

	if seed == 0 {

		rng_state = 88172645463393265
	} else {

		rng_state = seed
	}
}

rng_next_u64 :: #force_inline proc "contextless" ( ) ->
                                                  u64 {

	x := rng_state
	x ~= x >> 12
	x ~= x << 25
	x ~= x >> 27
	rng_state = x
	return x * 2685821657736338717
}

frand01 :: proc "contextless" ( ) ->
                               f32 {

	// Take top 24 bits -> [0,1)
	u := u32( rng_next_u64( ) >> 40 )
	return f32( u ) / f32( 1 << 24 )
}

// Box-Muller N(0,1)
frand_normal :: proc "contextless" ( ) ->
                                    f32 {

	u1 := frand01( )
	u2 := frand01( )
	if u1 < 1e-12 {

	    u1 = 1e-12
	}

	return f32( math.sqrt_f64( -2.0 * math.log( f64( u1 ), math.E ) ) *
                math.cos( 2.0 * math.PI * f64( u2 ) ) )
}

shuffle_int :: proc "contextless" ( a : [ ]int ) {

	// Fisher-Yates
	for i := len( a ) - 1; i > 0; i -= 1 {

     	j := int( rng_next_u64( ) % u64( i + 1 ) )
		tmp := a[ i ]
		a[ i ] = a[ j ]
		a[ j ] = tmp
	}
}

//
// Byte reader / writer ( for weights + MNIST )
//

Byte_Reader :: struct {

	data : [ ]u8,
	off  :  int,
}

br_need :: proc ( r : ^Byte_Reader,
                  n : int ) {

	if r.off + n > len( r.data ) {

	    die( "ERROR : Unexpected EOF while parsing file." )
	}
}

br_u8 :: proc ( r : ^Byte_Reader ) ->
                u8 {

	br_need( r, 1 )
	b     := r.data[ r.off ]
	r.off += 1
	return b
}

br_u32_le :: proc ( r : ^Byte_Reader ) ->
                    u32 {

	br_need( r, 4 )
	b0 := u32( r.data[ r.off + 0 ] )
	b1 := u32( r.data[ r.off + 1 ] )
	b2 := u32( r.data[ r.off + 2 ] )
	b3 := u32( r.data[ r.off + 3 ] )
	r.off += 4
	return ( b0 ) | ( b1 << 8 ) | ( b2 << 16 ) | ( b3 << 24 )
}

br_u32_be :: proc ( r : ^Byte_Reader ) ->
                    u32 {

	br_need( r, 4 )
	b0 := u32( r.data[ r.off + 0 ] )
	b1 := u32( r.data[ r.off + 1 ] )
	b2 := u32( r.data[ r.off + 2 ] )
	b3 := u32( r.data[ r.off + 3 ] )
	r.off += 4
	return ( b0 << 24 ) | ( b1 << 16 ) | ( b2 << 8 ) | ( b3 )
}

br_f32_le :: proc ( r : ^Byte_Reader ) ->
                    f32 {

	u := br_u32_le( r )
	return transmute( f32 )u
}

br_bytes_view :: proc ( r : ^Byte_Reader,
                        n : int ) ->
                        [ ]u8 {

	br_need( r, n )
	s     := r.data[ r.off : r.off + n ]
	r.off += n
	return s
}

Byte_Writer :: struct {

	buf: [ dynamic ]u8
}

bw_u8 :: proc ( w : ^Byte_Writer,
                b : u8 ) {

	append( & w.buf, b )
}

bw_u32_le :: proc ( w : ^Byte_Writer,
                    v : u32 ) {

	bw_u8( w, u8( v & 0xff ) )
	bw_u8( w, u8( ( v >> 8 ) & 0xff ) )
	bw_u8( w, u8( ( v >> 16 ) & 0xff ) )
	bw_u8( w, u8( ( v >> 24 ) & 0xff ) )
}

bw_f32_le :: proc ( w : ^Byte_Writer,
                    v : f32 ) {

	u := transmute( u32 )v
	bw_u32_le( w, u )
}

bw_f32_slice_le :: proc ( w : ^Byte_Writer,
                          s : [ ]f32 ) {

	for i in 0 ..< len( s ) {

	    bw_f32_le( w, s[ i ] )
	}
}

//
// MNIST IDX loader ( uncompressed )
//

Mnist :: struct {

	n      : int,
    rows   : int,
    cols   : int,
	images : [ ]u8, // n * rows * cols
	labels : [ ]u8, // n
}

mnist_free :: proc ( m : ^Mnist ) {

	if m == nil do return
	if m.images != nil do delete( m.images )
	if m.labels != nil do delete( m.labels )
	m^ = Mnist{ }
}

mnist_load_images_idx :: proc ( path     : string,
                                out_n    : ^int,
                                out_rows : ^int,
                                out_cols : ^int ) ->
                                [ ]u8 {

	data, ok := os.read_entire_file( path )
	if !ok do die2( "ERROR : Could not open ", path )

	r := Byte_Reader{

	    data = data,
		off  = 0
	}

	magic := br_u32_be( & r )
	if magic != 2051 {

		fmt.printf( "ERROR : Bad magic in images file %s ( got %d, expected 2051 )\n",
	                path, magic )
		os.exit( -1 )
	}

	n    := int( br_u32_be( & r ) )
	rows := int( br_u32_be( & r ) )
	cols := int( br_u32_be( & r ) )

	total   := n * rows * cols
	payload := br_bytes_view( & r, total )

	buf := make( [ ]u8, total )
	copy( buf, payload )

	delete( data )

	out_n^    = n
	out_rows^ = rows
	out_cols^ = cols
	return buf
}

mnist_load_labels_idx :: proc ( path  : string,
                                out_n : ^int ) ->
                                [ ]u8 {

	data, ok := os.read_entire_file( path )
	if !ok do die2( "ERROR : Could not open ", path )

	r := Byte_Reader{

	    data = data,
		off  = 0
	}

	magic := br_u32_be( & r )
	if magic != 2049 {

		fmt.printf( "ERROR : Bad magic in labels file %s ( got %d, expected 2049 )\n",
	                path, magic )
		os.exit( -1 )
	}

	n := int( br_u32_be( & r ) )
	payload := br_bytes_view( & r, n )

	buf := make( [ ]u8, n )
	copy( buf, payload )

	delete( data )

	out_n^ = n
	return buf
}

//
// Layer_Norm ( no affine )
//

LN_Cache :: struct {

	mean   : f32,
	invstd : f32,
}

layer_norm_forward :: proc "contextless" ( x : [ ]f32,
                                           y : [ ]f32,
                                           c : ^LN_Cache ) {

	n    := len( x )
	mean : f32 = 0
	for i in 0 ..< n {

	    mean += x[ i ]
	}
	mean /= f32( n )

	var : f32 = 0
	for i in 0 ..< n {

	    d := x[ i ] - mean
		var += d * d
	}
	var /= f32( n )

	invstd := 1.0 / f32( math.sqrt( f64( var + BLA_EPS ) ) )

	for i in 0 ..< n {

	    y[ i ] = ( x[ i ] - mean ) * invstd
	}

	c.mean = mean
	c.invstd = invstd
}

layer_norm_backward :: proc "contextless" (
                            dout   : [ ]f32,
                            y      : [ ]f32,
                            invstd : f32,
                            dx     : [ ]f32 ) {

	n := len( dout )

	sum1 : f32 = 0
	sum2 : f32 = 0
	for i in 0 ..< n {

		sum1 += dout[ i ]
		sum2 += dout[ i ] * y[ i ]
	}
	invn := 1.0 / f32( n )
	for i in 0 ..< n {

		dx[ i ] = ( invstd * invn ) * ( f32( n ) * dout[ i ] - sum1 - y[ i ] * sum2 )
	}
}

//
// Bit_Linear layer ( ternary W, int8 activations )
//

Bit_Linear :: struct {

	in_dim  : int,
    out_dim : int,

	W       : [ ]f32,   // [ out_dim * in_dim ]
	b       : [ ]f32,   // [ out_dim ]

	mW      : [ ]f32,
    vW      : [ ]f32,
	mb      : [ ]f32,
    vb      : [ ]f32,

	Wt      : [ ]i8,    // ternary { -1, 0, +1 }
	beta    : f32
}

bit_linear_init :: proc ( L               : ^Bit_Linear,
                          in_dim, out_dim : int ) {

	L^        = Bit_Linear{ }
	L.in_dim  = in_dim
	L.out_dim = out_dim

	wcount := in_dim * out_dim

	L.W  = make( [ ]f32, wcount )
	L.b  = make( [ ]f32, out_dim )

	L.mW = make( [ ]f32, wcount )
	L.vW = make( [ ]f32, wcount )
	L.mb = make( [ ]f32, out_dim )
	L.vb = make( [ ]f32, out_dim )

	L.Wt = make( [ ]i8, wcount )

	std := f32( math.sqrt( 2.0 / f64( in_dim ) ) )

	for i in 0 ..< wcount {

		L.W[ i ]  = std * frand_normal( )
		L.mW[ i ] = 0
		L.vW[ i ] = 0
		L.Wt[ i ] = 0
	}

	for o in 0 ..< out_dim {

		L.b[ o ]  = 0
		L.mb[ o ] = 0
		L.vb[ o ] = 0
	}

	L.beta = 1
}

bit_linear_free :: proc ( L : ^Bit_Linear ) {

	if L.W  != nil do delete( L.W )
	if L.b  != nil do delete( L.b )
	if L.mW != nil do delete( L.mW )
	if L.vW != nil do delete( L.vW )
	if L.mb != nil do delete( L.mb )
	if L.vb != nil do delete( L.vb )
	if L.Wt != nil do delete( L.Wt )

	L^ = Bit_Linear{ }
}

bit_linear_quantize_weights :: proc ( L : ^Bit_Linear ) {

	wcount := len( L.W )

	sum_abs : f64 = 0
	for i in 0 ..< wcount {

	    sum_abs += math.abs( f64( L.W[ i ] ) )
	}

	beta := f32( sum_abs / f64( wcount ) )
	if beta < 1e-12 {

	    beta = 1e-12
	}
	L.beta = beta

	for i in 0 ..< wcount {

		w := L.W[ i ] / beta

		// jnc
		// q := lrintf_even(w)
		// q = clampi(q, -1, 1)

		// q := int( w )
		// q = q < -0.5 ? -1 : q
		// q = q > 1 ? 1 : q

		q : i8 = 0

		if w > 0.5 {

		    q = 1
		}

	    if w < -0.5 {

		    q = -1
		}

		// L.Wt[ i ] = i8( q )
		L.Wt[ i ] = q
	}
}

bit_linear_forward_quantized :: proc "contextless" (
                                     L          : ^Bit_Linear,
                                     x_ln       : [ ]f32,
                                     xq_scratch : [ ]i8,
                                     y_out      : [ ]f32 ) {
	in_dim  := L.in_dim
	out_dim := L.out_dim

	// gamma = absmax(x_ln)
	gamma : f32 = 0
	for i in 0 ..< in_dim {

	    // a := f32(math.abs(f64(x_ln[i])))

		a := abs( x_ln[ i ] )

		if a > gamma {

		    gamma = a
		}
	}
	if gamma < 1e-12 {

	    gamma = 1e-12
	}

	s_in := f32( QP ) / gamma

	for i in 0 ..< in_dim {
		// jnc
	    // q := lrintf_even(x_ln[i] * s_in)
		// q = clampi(q, QN, QP)

		q := int( x_ln[ i ] * s_in )

		if q < QN {

		    q = QN
		} else if q > QP {

		    q = QP
		}

		xq_scratch[ i ] = i8( q )
	}

	s_out := ( L.beta * gamma ) / f32( QP )

	for o in 0 ..< out_dim {

		acc     : i32 = 0
		row_off := o * in_dim

		// jnc
		tmp := L.Wt[ row_off + 0 : row_off + in_dim ]

		for i in 0 ..< in_dim {
			// w := i32( L.Wt[ row_off + i ] )     // -1, 0, +1

			// w := i32( tmp[ i ] )                // -1,0,+1
			// xq := i32( xq_scratch[ i ] )        // int8
			// acc += xq * w

			acc += i32( xq_scratch[ i ] * tmp[ i ] )
		}

/*

        for i in 0 ..< in_dim / 16 {

            v := i * 16

			acc_1 := i32( xq_scratch[ v ]     * tmp[ v ] )
			acc_2 := i32( xq_scratch[ v + 1 ] * tmp[ v + 1 ] )
			acc_3 := i32( xq_scratch[ v + 2 ] * tmp[ v + 2 ] )
			acc_4 := i32( xq_scratch[ v + 3 ] * tmp[ v + 3 ] )

			acc_5 := i32( xq_scratch[ v + 4 ] * tmp[ v + 4 ] )
			acc_6 := i32( xq_scratch[ v + 5 ] * tmp[ v + 5 ] )
			acc_7 := i32( xq_scratch[ v + 6 ] * tmp[ v + 6 ] )
			acc_8 := i32( xq_scratch[ v + 7 ] * tmp[ v + 7 ] )

			acc_9  := i32( xq_scratch[ v + 8 ]  * tmp[ v + 8 ] )
			acc_10 := i32( xq_scratch[ v + 9 ]  * tmp[ v + 9 ] )
			acc_11 := i32( xq_scratch[ v + 10 ] * tmp[ v + 10 ] )
			acc_12 := i32( xq_scratch[ v + 11 ] * tmp[ v + 11 ] )


			acc_13 := i32( xq_scratch[ v + 12 ] * tmp[ v + 12 ] )
			acc_14 := i32( xq_scratch[ v + 13 ] * tmp[ v + 13 ] )
			acc_15 := i32( xq_scratch[ v + 14 ] * tmp[ v + 14 ] )
			acc_16 := i32( xq_scratch[ v + 15 ] * tmp[ v + 15 ] )


			acc_1 += acc_2 + acc_3 + acc_4

			acc_5 += acc_6 + acc_7 + acc_8

			acc_9 += acc_10 + acc_11 + acc_12

			acc_13 += acc_14 + acc_15 + acc_16

			acc += acc_1 + acc_5 + acc_9 + acc_13
		}

		bb := ( in_dim / 16 ) * 16

		for i in 0 ..< in_dim % 16 {

			acc += i32( xq_scratch[ bb + i ] * tmp[ bb + i ] )
		}

*/


		y_out[ o ] = f32( acc ) * s_out + L.b[ o ]
	}

	// assert( my_in_dim == in_dim )
	// fmt.printfln( "==> [ %v ]", in_dim )
}

// STE backward : approximate as FP32 linear layer
bit_linear_backward_ste :: proc "contextless" (
                                L         : ^Bit_Linear,
                                x_ln      : [ ]f32,
                                dy        : [ ]f32,
                                dW_accum  : [ ]f32,
                                db_accum  : [ ]f32,
                                dx_ln_out : [ ]f32 ) {

	in_dim  := L.in_dim
	out_dim := L.out_dim

	for o in 0 ..< out_dim {

		g := dy[ o ]
		db_accum[ o ] += g
		row_off := o * in_dim
		for i in 0 ..< in_dim {

	        dW_accum[ row_off + i ] += g * x_ln[ i ]
		}
	}

	for i in 0..<in_dim {

		sum : f32 = 0
		for o in 0 ..< out_dim {

			sum += L.W[ o * in_dim + i ] * dy[ o ]
		}

		dx_ln_out[ i ] += sum
	}
}

adam_update :: proc "contextless" (
                            param : [ ]f32,
                            m     : [ ]f32,
                            v     : [ ]f32,
                            grad  : [ ]f32,
                            lr    : f32,
                            beta1 : f32,
                            beta2 : f32,
                            eps   : f32,
                            t     : int ) {

	b1t := 1.0 - f32( math.pow( f64( beta1 ), f64( t ) ) )
	b2t := 1.0 - f32( math.pow( f64( beta2 ), f64( t ) ) )

	for i in 0 ..< len( param ) {

		g     := grad[ i ]
		m[ i ] = beta1 * m[ i ] + ( 1 - beta1 ) * g
		v[ i ] = beta2 * v[ i ] + ( 1 - beta2 ) * ( g * g )
		mhat := m[ i ] / b1t
		vhat := v[ i ] / b2t
		// jnc
		param[ i ] -= lr * mhat / ( f32( math.sqrt( f64( vhat ) ) ) + eps )

		// param[ i ] -= lr * mhat / ( math.sqrt_f32( vhat ) + eps )
	}
}

//
// Activations: ReLU^2
//

relu2_forward :: proc "contextless" (
                        x : [ ]f32,
                        y : [ ]f32 ) {

	for i in 0 ..< len( x ) {

		v := x[ i ]
		y[ i ] = ( v > 0 ) ? ( v * v ) : 0
	}
}

relu2_backward :: proc "contextless" (
                        x_pre : [ ]f32,
                        dy    : [ ]f32,
                        dx    : [ ]f32 ) {

	for i in 0 ..< len( x_pre ) {

		v := x_pre[ i ]
		dx[ i ] = ( v > 0 ) ? ( dy[ i ] * 2.0 * v ) : 0
	}
}

//
// Softmax + cross-entropy
//

softmax_ce_loss_and_grad :: proc "contextless" (
                                logits  : [ ]f32,
                                y_true  : int,
                                dlogits : [ ]f32 ) ->
                                f32 {

	nclass := len( logits )

	m := logits[ 0 ]
	for i in 1 ..< nclass {

		if logits[ i ] > m {

	        m = logits[ i ]
		}
	}

	sum: f32 = 0
	for i in 0 ..< nclass {

		e := f32( math.exp( f64( logits[ i ] - m ) ) )
		dlogits[ i ] = e
		sum += e
	}
	invsum := 1.0 / sum
	for i in 0 ..< nclass {

	    dlogits[ i ] *= invsum
	}

	p := dlogits[ y_true ]
	loss := -f32( math.log( f64( p + 1e-12 ), math.E ) )

	for i in 0 ..< nclass {

		dlogits[ i ] = dlogits[ i ] - ( ( i == y_true ) ? 1.0 : 0.0 )
	}

	return loss
}

softmax_probs :: proc "contextless" (
                        logits : [ ]f32,
                        probs  : [ ]f32 ) {

	n := len( logits )
	m := logits[ 0 ]
	for i in 1 ..< n {

		if logits[ i ] > m {

		    m = logits[ i ]
		}
	}

	sum: f32 = 0
	for i in 0 ..< n {

		e := f32( math.exp( f64( logits[ i ] - m ) ) )
		probs[ i ] = e
		sum       += e
	}
	inv : f32 = 1.0 / ( sum + 1e-20 )
	for i in 0 ..< n {

	    probs[ i ] *= inv
	}
}

arg_max_f :: proc ( x : [ ]f32 ) ->
                    int {

	best := 0
	bv   := x[ 0 ]
	for i in 1 ..< len( x ) {

		if x[ i ] > bv {

			bv = x[ i ]
			best = i
		}
	}

	return best
}

//
// Network
//

Net :: struct {

	input_dim : int,
	hidden    : int,
	nclass    : int,

	l1        : Bit_Linear,
    l2        : Bit_Linear,
    l3        : Bit_Linear,
	step      : int
}

net_init :: proc ( net    : ^Net,
                   hidden : int ) {

	net.input_dim = 28*28
	net.hidden    = hidden
	net.nclass    = 10
	net.step      = 0
	bit_linear_init( & net.l1, net.input_dim, hidden )
	bit_linear_init( & net.l2, hidden, hidden )
	bit_linear_init( & net.l3, hidden, net.nclass )
}

net_free :: proc ( net : ^Net ) {

	bit_linear_free( & net.l1 )
	bit_linear_free( & net.l2 )
	bit_linear_free( & net.l3 )

	net^ = Net{ }
}

net_quantize_all :: proc( net : ^Net ) {

	bit_linear_quantize_weights( & net.l1 )
	bit_linear_quantize_weights( & net.l2 )
	bit_linear_quantize_weights( & net.l3 )
}

Workspace :: struct {

	x0      : [ ]f32,
    x0_ln   : [ ]f32,
	ln1     : LN_Cache,

	h1_pre  : [ ]f32,
    h1      : [ ]f32,
    h1_ln   : [ ]f32,
	ln2     : LN_Cache,

	h2_pre  : [ ]f32,
    h2      : [ ]f32,
    h2_ln   : [ ]f32,
	ln3     : LN_Cache,

	logits  : [ ]f32,

	xq      : [ ]i8,

	dlogits : [ ]f32,

	dh2_ln  : [ ]f32,
    dh2     : [ ]f32,
    dh2_pre : [ ]f32,

    dh1_ln  : [ ]f32,
    dh1     : [ ]f32,
    dh1_pre : [ ]f32,

    dx0_ln  : [ ]f32,
    dx0     : [ ]f32
}

workspace_init :: proc ( ws     : ^Workspace,
                         hidden : int ) {

    ws^       = Workspace{ }

	ws.x0     = make( [ ]f32, 784 )
	ws.x0_ln  = make( [ ]f32, 784 )

	ws.h1_pre = make( [ ]f32, hidden )
	ws.h1     = make( [ ]f32, hidden )
	ws.h1_ln  = make( [ ]f32, hidden )

	ws.h2_pre = make( [ ]f32, hidden )
	ws.h2     = make( [ ]f32, hidden )
	ws.h2_ln  = make( [ ]f32, hidden )

	ws.logits = make( [ ]f32, 10 )

	maxdim := hidden
	if 784 > maxdim {

	    maxdim = 784
	}
	ws.xq      = make( [ ]i8, maxdim )

	ws.dlogits = make( [ ]f32, 10 )

	ws.dh2_ln  = make( [ ]f32, hidden )
	ws.dh2     = make( [ ]f32, hidden )
	ws.dh2_pre = make( [ ]f32, hidden )

	ws.dh1_ln  = make( [ ]f32, hidden )
	ws.dh1     = make( [ ]f32, hidden )
	ws.dh1_pre = make( [ ]f32, hidden )

	ws.dx0_ln  = make( [ ]f32, 784 )
	ws.dx0     = make( [ ]f32, 784 )
}

workspace_free :: proc ( ws : ^Workspace ) {

	if ws.x0    != nil do delete( ws.x0 )
	if ws.x0_ln != nil do delete( ws.x0_ln )

	if ws.h1_pre != nil do delete( ws.h1_pre )
	if ws.h1     != nil do delete( ws.h1 )
	if ws.h1_ln  != nil do delete( ws.h1_ln )

	if ws.h2_pre != nil do delete( ws.h2_pre )
	if ws.h2     != nil do delete( ws.h2 )
	if ws.h2_ln  != nil do delete( ws.h2_ln )

	if ws.logits != nil do delete( ws.logits )
	if ws.xq     != nil do delete( ws.xq )

	if ws.dlogits != nil do delete( ws.dlogits )

	if ws.dh2_ln  != nil do delete( ws.dh2_ln )
	if ws.dh2     != nil do delete( ws.dh2 )
	if ws.dh2_pre != nil do delete( ws.dh2_pre )

	if ws.dh1_ln  != nil do delete( ws.dh1_ln )
	if ws.dh1     != nil do delete( ws.dh1 )
	if ws.dh1_pre != nil do delete( ws.dh1_pre )

	if ws.dx0_ln  != nil do delete( ws.dx0_ln )
	if ws.dx0     != nil do delete( ws.dx0 )

	ws^ = Workspace{ }
}

net_forward_sample_quantized :: proc(
                    net          : ^Net,
                    ws           : ^Workspace,
                    img          : [ ]u8,
                    label        : u8,
                    compute_loss : bool ) ->
                    f32 {

	for i in 0 ..< 784 {

	    ws.x0[ i ] = f32( img[ i ] ) / 255.0
	}

	layer_norm_forward( ws.x0, ws.x0_ln, & ws.ln1 )
	bit_linear_forward_quantized( & net.l1, ws.x0_ln, ws.xq, ws.h1_pre )
	relu2_forward( ws.h1_pre, ws.h1 )

	layer_norm_forward( ws.h1, ws.h1_ln, & ws.ln2 )
	bit_linear_forward_quantized( & net.l2, ws.h1_ln, ws.xq, ws.h2_pre )
	relu2_forward( ws.h2_pre, ws.h2 )

	layer_norm_forward( ws.h2, ws.h2_ln, & ws.ln3 )
	bit_linear_forward_quantized( & net.l3, ws.h2_ln, ws.xq, ws.logits )

	if !compute_loss {

	    return 0
	}

	return softmax_ce_loss_and_grad( ws.logits, int( label ), ws.dlogits )
}

net_backward_sample_ste :: proc (
                                net : ^Net,
                                ws  : ^Workspace,
                                dW1 : [ ]f32,
                                db1 : [ ]f32,
                                dW2 : [ ]f32,
                                db2 : [ ]f32,
                                dW3 : [ ]f32,
                                db3 : [ ]f32 ) {

	H := net.hidden

	#force_inline zeros_f32( ws.dh2_ln )
	#force_inline zeros_f32( ws.dh2 )
	#force_inline zeros_f32( ws.dh2_pre )

	#force_inline zeros_f32( ws.dh1_ln )
	#force_inline zeros_f32( ws.dh1 )
	#force_inline zeros_f32( ws.dh1_pre )

	#force_inline zeros_f32( ws.dx0_ln )
	#force_inline zeros_f32( ws.dx0 )

	bit_linear_backward_ste( & net.l3, ws.h2_ln, ws.dlogits, dW3, db3, ws.dh2_ln )
	layer_norm_backward( ws.dh2_ln, ws.h2_ln, ws.ln3.invstd, ws.dh2 )
	relu2_backward( ws.h2_pre, ws.dh2, ws.dh2_pre )

	bit_linear_backward_ste( & net.l2, ws.h1_ln, ws.dh2_pre, dW2, db2, ws.dh1_ln )
	layer_norm_backward( ws.dh1_ln, ws.h1_ln, ws.ln2.invstd, ws.dh1 )
	relu2_backward( ws.h1_pre, ws.dh1, ws.dh1_pre )

	bit_linear_backward_ste( & net.l1, ws.x0_ln, ws.dh1_pre, dW1, db1, ws.dx0_ln )
	layer_norm_backward( ws.dx0_ln, ws.x0_ln, ws.ln1.invstd, ws.dx0 )
}

//
// Weights save / load ( BLA1 )
//

bit_linear_save :: proc ( L : ^Bit_Linear,
                          w : ^Byte_Writer ) {

    bw_u32_le( w, u32( L.in_dim ) )
	bw_u32_le( w, u32( L.out_dim ) )
	bw_f32_slice_le( w, L.W )
	bw_f32_slice_le( w, L.b )
}

bit_linear_load :: proc ( L : ^Bit_Linear,
                          r : ^Byte_Reader ) {

	in_dim  := int( br_u32_le( r ) )
	out_dim := int( br_u32_le( r ) )

	if in_dim != L.in_dim ||
       out_dim != L.out_dim {

        die( "ERROR : Weights file dims don't match the constructed network." )
    }

	for i in 0 ..< len( L.W ) {

	    L.W[i] = br_f32_le( r )
	}
	for i in 0 ..< len( L.b ) {

	    L.b[ i ] = br_f32_le( r )
	}

	zeros_f32( L.mW )
	zeros_f32( L.vW )
	zeros_f32( L.mb )
	zeros_f32( L.vb )
}

net_save :: proc ( net  : ^Net,
                   path : string ) {

	w := Byte_Writer{ }

	// w.buf = make( [ dynamic ]u8, len = 0, cap=10*1024*1024  )

	bw_u32_le( & w, BLA_MAGIC )
	bw_u32_le( & w, 1 ) // version
	bw_u32_le( & w, u32( net.input_dim ) )
	bw_u32_le( & w, u32( net.hidden ) )
	bw_u32_le( & w, u32( net.nclass ) )
	bw_u32_le( & w, u32( net.step ) )

	bit_linear_save( & net.l1, & w )
	bit_linear_save( & net.l2, & w )
	bit_linear_save( & net.l3, & w )

	ok := os.write_entire_file( path, w.buf[ : ] )
	if !ok {

	    die2( "ERROR : Could not open for writing : ", path )
	}

	delete( w.buf )
}

net_load :: proc ( net  : ^Net,
                   path : string ) ->
                   bool {

	data, ok := os.read_entire_file( path )
	if !ok {

	    return false
	}
	r := Byte_Reader{

	    data = data,
		off  = 0
	}

	magic := br_u32_le( & r )
	if magic != BLA_MAGIC {

	    die( "ERROR : Bad weights file magic." )
	}

	ver := br_u32_le( & r )
	if ver != 1 {

	    die( "ERROR : Unsupported weights file version." )
	}

	input_dim := int( br_u32_le( & r ) )
	hidden    := int( br_u32_le( & r ) )
	nclass    := int( br_u32_le( & r ) )
	step      := int( br_u32_le( & r ) )

	if input_dim != 784 ||
       nclass    != 10 {

           die( "ERROR : Weights file isn't for MNIST 784->...->10 ." )
    }

	if net.hidden == 0 {

	    net_init( net, hidden )
	}
	if net.hidden != hidden {

	    die( "ERROR : Hidden size mismatch between args and weights file." )
	}

	net.step = step
	bit_linear_load( & net.l1, & r )
	bit_linear_load( & net.l2, & r )
	bit_linear_load( & net.l3, & r )

	delete( data )
	return true
}

//
// Multi_threaded training / eval
//

Thread_Ctx :: struct {

	ws       : Workspace,

	dW1      : [ ]f32,
    dW2      : [ ]f32,
    dW3      : [ ]f32,

	db1      : [ ]f32,
    db2      : [ ]f32,
    db3      : [ ]f32,

	loss_sum : f64,
	correct  : int
}

thread_ctx_init :: proc ( tc  : ^Thread_Ctx,
                          net : ^Net ) {

	H := net.hidden
	workspace_init( & tc.ws, H )

	w1 := net.l1.in_dim * net.l1.out_dim
	w2 := net.l2.in_dim * net.l2.out_dim
	w3 := net.l3.in_dim * net.l3.out_dim

	tc.dW1 = make( [ ]f32, w1 )
	tc.dW2 = make( [ ]f32, w2 )
	tc.dW3 = make( [ ]f32, w3 )

	tc.db1 = make( [ ]f32, net.l1.out_dim )
	tc.db2 = make( [ ]f32, net.l2.out_dim )
	tc.db3 = make( [ ]f32, net.l3.out_dim )

	tc.loss_sum = 0
	tc.correct  = 0
}

thread_ctx_free :: proc ( tc : ^Thread_Ctx ) {

	workspace_free( & tc.ws )

	if tc.dW1 != nil do delete( tc.dW1 )
	if tc.dW2 != nil do delete( tc.dW2 )
	if tc.dW3 != nil do delete( tc.dW3 )
	if tc.db1 != nil do delete( tc.db1 )
	if tc.db2 != nil do delete( tc.db2 )
	if tc.db3 != nil do delete( tc.db3 )

	tc^ = Thread_Ctx{ }
}

thread_ctx_zero_grads :: proc ( tc : ^Thread_Ctx ) {

	#force_inline zeros_f32( tc.dW1 )
	#force_inline zeros_f32( tc.dW2 )
	#force_inline zeros_f32( tc.dW3 )
	#force_inline zeros_f32( tc.db1 )
	#force_inline zeros_f32( tc.db2 )
	#force_inline zeros_f32( tc.db3 )

	tc.loss_sum = 0
	tc.correct  = 0
}

Train_Job :: struct {

	net       : ^Net,
	data      : ^Mnist,
	perm      : [ ]int,
	start     : int,
	bs        : int,
	thread_id : int,
	nthreads  : int,
	tcs       : [ ]Thread_Ctx
}

Eval_Job :: struct {

	net       : ^Net,
	data      : ^Mnist,
	limit     : int,
	thread_id : int,
	nthreads  : int,
	tcs       : [ ]Thread_Ctx
}

train_thread_fn :: proc "c" ( arg : rawptr ) ->
                              rawptr {

	// Ensure context is valid on this OS thread
	context = runtime.default_context( )

	job   := ( ^Train_Job )( arg )
	net   := job.net
	train := job.data
	tc    := & job.tcs[ job.thread_id ]

	per := ( job.bs + job.nthreads - 1 ) / job.nthreads
	lo  := job.thread_id * per
	hi  := lo + per
	if hi > job.bs {

	    hi = job.bs
	}

	img_stride := train.rows * train.cols

	for bi in lo ..< hi {

		idx := job.perm[ job.start + bi ]
		img := train.images[ idx * img_stride : ( idx + 1 ) * img_stride ]
		y   := train.labels[ idx ]

		loss := net_forward_sample_quantized( net, & tc.ws, img, y, true )

		tc.loss_sum += f64( loss )

		net_backward_sample_ste( net,
	                             & tc.ws,
								 tc.dW1, tc.db1, tc.dW2,
								 tc.db2, tc.dW3, tc.db3 )
	}
	return nil
}

eval_thread_fn :: proc "c" ( arg : rawptr ) ->
                             rawptr {

	context = runtime.default_context()

	job  := ( ^Eval_Job )( arg )
	net  := job.net
	data := job.data
	tc   := & job.tcs[ job.thread_id ]

	n := data.n
	if job.limit > 0 &&
       job.limit < n {

        n = job.limit
    }

	per := ( n + job.nthreads - 1 ) / job.nthreads
	lo  := job.thread_id * per
	hi  := lo + per
	if hi > n {

	    hi = n
	}

	img_stride := data.rows * data.cols
	correct    := 0

	for idx in lo ..< hi {

		img := data.images[ idx * img_stride : ( idx + 1 ) * img_stride ]
		y   := data.labels[ idx ]
		_    = net_forward_sample_quantized( net, & tc.ws, img, y, false )
		pred := arg_max_f( tc.ws.logits )
		if pred == int( y ) {

		    correct += 1
		}
	}

	tc.correct = correct
	return nil
}

evaluate_accuracy_mt :: proc ( net      : ^Net,
                               test     : ^Mnist,
                               limit    : int,
                               nthreads : int,
                               tcs      : [ ]Thread_Ctx ) ->
                               f32 {

    nthreads := nthreads
	if nthreads < 1 {

	    nthreads = 1
	}

	net_quantize_all( net )

	n := test.n
	if limit > 0 && limit < n {

	    n = limit
	}

	if nthreads == 1 {

		correct    := 0
		img_stride := test.rows * test.cols
		for idx in 0 ..< n {

			img := test.images[ idx * img_stride : ( idx + 1 ) * img_stride ]
			y   := test.labels[ idx ]
			_ = net_forward_sample_quantized( net, & tcs[ 0 ].ws, img, y, false )
			if arg_max_f( tcs[ 0 ].ws.logits ) == int( y ) {

		        correct += 1
			}
		}
		return f32( correct ) / f32( n )
	}

	jobs := make( [ ]Eval_Job, nthreads )
	ths  := make( [ ]posix.pthread_t, nthreads )

	for t in 0 ..< nthreads {

		tcs[ t ].correct = 0
		jobs[ t ] = Eval_Job{

		    net       = net,
			data      = test,
		    limit     = n,
			thread_id = t,
		    nthreads  = nthreads,
			tcs       = tcs,
		}
		rc : posix.Errno = posix.pthread_create( & ths[ t ], nil, eval_thread_fn, rawptr( & jobs[ t ] ) )
		if rc != .NONE {

	        die( "ERROR : pthread_create failed." )
		}
	}

	for t in 0 ..< nthreads {

		_ = posix.pthread_join( ths[ t ], nil )
	}

	correct := 0
	for t in 0 ..< nthreads {

	    correct += tcs[ t ].correct
	}

	delete( jobs )
	delete( ths )

	return f32( correct ) / f32( n )
}

//
// Command Line parsing
//

Args :: struct {

	train_images   : string,
	train_labels   : string,
	test_images    : string,
	test_labels    : string,

	epochs         : int,
	batch          : int,
	hidden         : int,
	lr             : f32,
	lr_start       : f32,
	lr_stop        : f32,
	limit_train    : int,
	limit_test     : int,
	seed           : u64,
	threads        : int,

	weights_load   : string,
	weights_save   : string,

	infer_index    : int,
	do_infer       : bool,
	infer_on_train : bool       // false = test,    true = train
}

args_defaults :: proc ( ) ->
                      Args {

	now_seed := u64( time.time_to_unix( time.now( ) ) )

	return Args{

		train_images   = "mnist_idx/train-images-idx3-ubyte",
		train_labels   = "mnist_idx/train-labels-idx1-ubyte",
		test_images    = "mnist_idx/t10k-images-idx3-ubyte",
		test_labels    = "mnist_idx/t10k-labels-idx1-ubyte",

		epochs         = 3,
		batch          = 64,
		hidden         = 256,
		lr             = 0.001, // or 0.005
		lr_start       = 0.01,
		lr_stop        = 0.00025,
		limit_train    = 0,
		limit_test     = 0,
		seed           = now_seed,
		threads        = 24,

		weights_load   = "",
		weights_save   = "weights.bla",

		infer_index    = -1,
		do_infer       = false,
		infer_on_train = false,
	}
}

print_help :: proc ( prog : string ) {

	fmt.printfln( "Usage: %s [options]\n", prog )
	fmt.println( "MNIST paths (IDX ubyte, uncompressed):" )
	fmt.println( "  --train-images PATH" )
	fmt.println( "  --train-labels PATH" )
	fmt.println( "  --test-images PATH" )
	fmt.println( "  --test-labels PATH\n" )
	fmt.println( "Training options:" )
	fmt.println( "  --epochs N        (default 3)" )
	fmt.println( "  --batch N         (default 64)" )
	fmt.println( "  --hidden N        (default 256)" )
	fmt.println( "  --lr LR           (default 0.001)" )
	fmt.println( "  --lr_start LR     (default 0.01)" )
	fmt.println( "  --lr_stop LR      (default 0.00025)" )
	fmt.println( "  --threads N       (default 24)" )
	fmt.println( "  --limit-train N   (0=all)" )
	fmt.println( "  --limit-test N    (0=all)" )
	fmt.println( "  --seed U\n")
	fmt.println( "Weights:")
	fmt.println( "  --weights PATH    Load weights (for inference or resume training)" )
	fmt.println( "  --save PATH       Save trained weights (default weights.bla)\n" )
	fmt.println( "Inference:")
	fmt.println( "  --infer-index N   Predict image N (requires --weights)" )
	fmt.println( "  --infer-split S   'test' (default) or 'train'" )
}

// Minimal numeric parsers for CLI (base10)

parse_i64 :: proc ( s : string ) ->
                  ( i64,
                    bool ) {

    return #force_inline strconv.parse_i64( s, 10 )
}

parse_u64 :: proc ( s : string ) ->
                  ( u64,
                    bool ) {

    return #force_inline  strconv.parse_u64( s, 10 )
}

parse_f64_simple :: proc ( s : string ) ->
                         ( f64,
                           bool ) {

    return #force_inline  strconv.parse_f64( s )
}

parse_args :: proc ( argv : [ ]string ) ->
                     Args {

	a := args_defaults( )

	i := 1
	for i < len( argv ) {

		arg := argv[ i ]

		need1 :: proc( i    : ^int,
	                   argv : [ ]string,
					   arg  : string ) ->
		               string {

			if i^ + 1 >= len( argv ) {

				fmt.printf( "ERROR : Unknown or incomplete option: %s\nTry --help\n", arg )
				os.exit( -1 )
			}

			i^ += 1
			return argv[ i^ ]
		}

		switch arg {

		case "--train-images" : a.train_images = need1( & i, argv, arg )
		case "--train-labels" : a.train_labels = need1( & i, argv, arg )
		case "--test-images"  : a.test_images  = need1( & i, argv, arg )
		case "--test-labels"  : a.test_labels  = need1( & i, argv, arg )

		case "--epochs":
			s := need1( & i, argv, arg )
			v, ok := parse_i64(s)
		    if !ok {

				die( "ERROR : Bad --epochs value ." )
			}
			a.epochs = int( v )

		case "--batch":
			s := need1( & i, argv, arg )
			v, ok := parse_i64( s )
		    if !ok {

				die( "ERROR : Bad --batch value ." )
			}
			a.batch = int( v )

		case "--hidden":
			s := need1( & i, argv, arg )
			v, ok := parse_i64( s )
		    if !ok {

				die( "ERROR : Bad --hidden value ." )
			}
			a.hidden = int( v )

		case "--lr":
			s     := need1( & i, argv, arg )
			v, ok := parse_f64_simple( s )
		    if !ok {

				die( "ERROR : Bad --lr value ." )
			}
			a.lr   = f32( v )

		case "--lr_start":
			s     := need1( & i, argv, arg )
			v, ok := parse_f64_simple( s )
			if !ok {

			    die( "ERROR : Bad --lr_start value ." )
			}
			a.lr   = f32( v )

		case "--lr_stop":
			s     := need1( & i, argv, arg )
			v, ok := parse_f64_simple( s );
		    if !ok {

				die( "ERROR : Bad --lr_stop value ." )
			}
			a.lr   = f32( v )

		case "--threads":
			s := need1( & i, argv, arg )
			v, ok := parse_i64( s )
		    if !ok {

				die( "ERROR : Bad --threads value ." )
			}
			a.threads = int( v )

		case "--limit-train":
			s := need1( & i, argv, arg )
			v, ok := parse_i64( s )
		    if !ok {

				die( "ERROR : Bad --limit-train value ." )
			}
			a.limit_train = int( v )

		case "--limit-test":
			s := need1( & i, argv, arg )
			v, ok := parse_i64( s )
		    if !ok {

				die( "ERROR : Bad --limit-test value ." )
			}
			a.limit_test = int( v )

		case "--seed":
			s := need1( & i, argv, arg )
			v, ok := parse_u64( s )
		    if !ok {

				die( "ERROR : Bad --seed value ." )
			}
			a.seed = v

		case "--weights":
			a.weights_load = need1( & i, argv, arg )

		case "--save":
			a.weights_save = need1( & i, argv, arg )

		case "--infer-index":
			s := need1( & i, argv, arg )
			v, ok := parse_i64( s )
		    if !ok {

				die( "ERROR : Bad --infer-index value ." )
			}
			a.infer_index = int( v )
			a.do_infer = true

		case "--infer-split":
			s := need1( & i, argv, arg )
			if s == "train" {

				a.infer_on_train = true
			} else if s == "test" {

				a.infer_on_train = false
			} else {

				die( "ERROR : --infer-split must be 'train' or 'test' ." )
			}

		case "--help":
			print_help( argv[ 0 ] )
			os.exit( 0 )

		case "-h":
			print_help( argv[ 0 ] )
			os.exit( 0 )

		case:
			fmt.printf("ERROR : Unknown or incomplete option: %s\nTry --help\n", arg )
			os.exit( -1 )
		}

		i += 1
	}

	if a.epochs <= 0 {

	    a.epochs = 1
	}
	if a.batch  <= 0 {

	    a.batch  = 32
	}
	if a.hidden <= 0 {

	    a.hidden = 128
	}
	if a.lr <= 0 {

	    a.lr = 1e-3
	}
	if a.threads <= 0 {

	    a.threads = 1
	}

	return a
}

//
// Pretty-print MNIST image
//

print_mnist_ascii :: proc ( img  : [ ]u8,
                            rows : int,
                            cols : int ) {

	ramp := " .:-=+*#%@"
	for r in 0 ..< rows {

		line : [ dynamic ]u8
		for c in 0 ..< cols {

			v := img[ r * cols + c ]
			idx := int( ( int( v ) * 9 ) / 255 )
			ch := ramp[ idx ]
			append( & line, u8( ch ) )
			append( & line, u8( ch ) )
		}
		append( & line, u8( '\n' ) )
		fmt.print( string( line[ : ] ) )
		delete( line )
	}
}

log_lerp_lr :: proc ( lr_min : f32,
                      lr_max : f32,
                      t      : f32 ) ->
                      f32 {

    log_min := math.log( lr_min, math.E )
    log_max := math.log( lr_max, math.E )
    log_lr  := log_min + ( 1 - t ) * ( log_max - log_min )
    return math.exp( log_lr )
}

//
// Main
//

main :: proc ( ) {

    // test_ternary_encoding_decoding( )

	args := parse_args( os.args )
	rng_seed( args.seed )

	train := Mnist{ }
	test  := Mnist{ }

	// Load MNIST
	nimg := 0
	rows := 0
	cols := 0

	train.images = mnist_load_images_idx( args.train_images, & nimg, & rows, & cols )
	train.n      = nimg
	train.rows   = rows
	train.cols   = cols

	nlab := 0
	train.labels = mnist_load_labels_idx( args.train_labels, & nlab )
	if nlab != train.n  {

	    die( "ERROR : Train images / labels count mismatch." )
	}
	if train.rows != 28 || train.cols != 28 {

	    die( "ERROR : Expected 28 x 28 MNIST images." )
	}

	test.images = mnist_load_images_idx( args.test_images, & nimg, & rows, & cols )
	test.n      = nimg
	test.rows   = rows
	test.cols   = cols
	test.labels = mnist_load_labels_idx(args.test_labels, &nlab)
	if nlab != test.n {

	    die( "ERROR : Test images / labels count mismatch.")
	}
	if test.rows != 28 || test.cols != 28 {

	    die( "ERROR : Expected 28 x 28 MNIST images." )
	}

	ntrain := train.n
	ntest  := test.n
	if args.limit_train > 0 &&
       args.limit_train < ntrain {

	    ntrain = args.limit_train
	}
	if args.limit_test  > 0 &&
       args.limit_test  < ntest  {

	    ntest  = args.limit_test
	}

	fmt.printfln( "\nLoaded MNIST: train=%d ( of %d ), test=%d ( of %d )\n",
                  ntrain, train.n, ntest, test.n )

	// Inference-only mode
	if args.do_infer {

		if args.weights_load == "" {

		    die( "ERROR : Inference requires --weights PATH " )
		}
		if args.infer_index < 0 {

	        die( "ERROR : Inference requires --infer-index N >= 0" )
		}

		net := Net{ }
		if !net_load( & net, args.weights_load ) {

	        die( "ERROR : Could not read weights file." )
		}
		fmt.printfln( "Loaded weights from %s ( hidden=%d, step=%d )\n",
	                  args.weights_load, net.hidden, net.step )

		ds := & test
		if args.infer_on_train {

	        ds = & train
		}

		if args.infer_index >= ds.n {

	        die( "ERROR : infer-index out of range." )
		}

 		tc := Thread_Ctx{ }
		thread_ctx_init( & tc, & net )

		net_quantize_all( & net )

		img_stride := ds.rows * ds.cols
		img := ds.images[ args.infer_index*img_stride : ( args.infer_index + 1 ) * img_stride ]
		y := ds.labels[ args.infer_index ]

		_ = net_forward_sample_quantized( & net, & tc.ws, img, y, false )

		probs := make( [ ]f32, 10 )
		softmax_probs( tc.ws.logits, probs )
		pred := arg_max_f( probs )

		fmt.printf( "Infer split=%s  index=%d\n", args.infer_on_train ? "train" : "test", args.infer_index )
		fmt.printfln( "True label: %d", int( y ) )
		fmt.printfln( "Pred label: %d", pred )
		fmt.println( "Probabilities:" )
		for i in 0 ..< 10 {

		    fmt.printfln( "  %d: %.6f", i, probs[ i ] )
		}
		fmt.println( "\nImage ( ASCII ):" )
		print_mnist_ascii( img, ds.rows, ds.cols )

		delete( probs )
		thread_ctx_free( & tc )
		net_free( &net )

		mnist_free( & train )
		mnist_free( & test )
		return
	}

	// Training ( the user can train or use the weights of a pre train model )
	net := Net{ }
	net_init( & net, args.hidden )

	if args.weights_load != "" {

		if !net_load( & net, args.weights_load ) {

		    die( "ERROR : Could not open weights file for loading" )
		}
		fmt.printfln( "Loaded weights from %s (step=%d)",
	                  args.weights_load, net.step )
	}

	perm := make( [ ]int, ntrain )
	for i in 0 ..< ntrain {

	    perm[ i ] = i
	}

	w1 := net.l1.in_dim * net.l1.out_dim
	w2 := net.l2.in_dim * net.l2.out_dim
	w3 := net.l3.in_dim * net.l3.out_dim

	dW1 := make( [ ]f32, w1 )
	dW2 := make( [ ]f32, w2 )
	dW3 := make( [ ]f32, w3 )
	db1 := make( [ ]f32, net.l1.out_dim )
	db2 := make( [ ]f32, net.l2.out_dim )
	db3 := make( [ ]f32, net.l3.out_dim )

	nthreads := args.threads
	if nthreads > args.batch {

	    nthreads = args.batch
	}

	if nthreads < 1 {

	    nthreads = 1
	}

	tcs := make( [ ]Thread_Ctx, nthreads )
	for t in 0 ..< nthreads {

	    thread_ctx_init( & tcs[ t ], & net )
	}

	beta1 : f32 = 0.9
	beta2 : f32 = 0.999
	eps   : f32 = 1e-6

	fmt.printfln( "Training with %d threads\n", nthreads )

	invert :: #force_inline proc "contextless" (
	                               range_max : int,
                                   slice_tmp : [ ]f32,
								   inv_bs    : f32 ) {

     			for i in 0 ..< range_max {

      		        slice_tmp[ i ] *= inv_bs
     			}
	}

	for epoch in 1 ..= args.epochs {

		shuffle_int( perm )

		epoch_loss : f64 = 0
		seen := 0

		start := 0
		for start < ntrain {

			bs := args.batch
			if start + bs > ntrain {

			    bs = ntrain - start
			}

			use_threads := nthreads
			if use_threads > bs {

		        use_threads = bs
			}
			if use_threads < 1 {

		        use_threads = 1
			}

			net_quantize_all( & net )

			#force_inline zeros_f32( dW1 )
			#force_inline zeros_f32( dW2 )
		    #force_inline zeros_f32( dW3 )
			#force_inline zeros_f32( db1 )
		    #force_inline zeros_f32( db2 )
		    #force_inline zeros_f32( db3 )

			for t in 0 ..< use_threads {

		        thread_ctx_zero_grads( & tcs[ t ] )
			}

			if use_threads == 1 {

				img_stride := train.rows * train.cols
				for bi in 0 ..< bs {

					idx  := perm[ start + bi ]
					img  := train.images[ idx * img_stride : ( idx + 1 ) * img_stride ]
					y    := train.labels[ idx ]
					loss := net_forward_sample_quantized( & net, & tcs[ 0 ].ws, img, y, true )
					tcs[ 0 ].loss_sum += f64( loss )
					net_backward_sample_ste( & net, & tcs[ 0 ].ws,
				                             tcs[ 0 ].dW1, tcs[ 0 ].db1,
											 tcs[ 0 ].dW2, tcs[ 0 ].db2,
											 tcs[ 0 ].dW3, tcs[ 0 ].db3 )
				}
			} else {

				jobs := make( [ ]Train_Job, use_threads )
				ths  := make( [ ]posix.pthread_t, use_threads )

				for t in 0 ..< use_threads {

					jobs[ t ] = Train_Job{

						net       = & net,
						data      = & train,
						perm      = perm[ : ],
						start     = start,
						bs        = bs,
						thread_id = t,
						nthreads  = use_threads,
						tcs       = tcs[ : ],
					}

					rc : posix.Errno = posix.pthread_create( & ths[ t ], nil, train_thread_fn, rawptr( & jobs[ t ] ) )
					if rc != .NONE {

					    die( "ERROR : pthread_create failed." )
					}
				}

				for t in 0 ..< use_threads {

					_ = posix.pthread_join( ths[ t ], nil )
				}

				delete( jobs )
				delete( ths )
			}

			// Reduce ( SUM the results of all threds )
			batch_loss : f64 = 0
			for t in 0 ..< use_threads do batch_loss += tcs[ t ].loss_sum

			for t in 0 ..< use_threads {

				for i in 0 ..< w1             do dW1[ i ] += tcs[ t ].dW1[ i ]
				for i in 0 ..< w2             do dW2[ i ] += tcs[ t ].dW2[ i ]
				for i in 0 ..< w3             do dW3[ i ] += tcs[ t ].dW3[ i ]
				for i in 0 ..< net.l1.out_dim do db1[ i ] += tcs[ t ].db1[ i ]
				for i in 0 ..< net.l2.out_dim do db2[ i ] += tcs[ t ].db2[ i ]
				for i in 0 ..< net.l3.out_dim do db3[ i ] += tcs[ t ].db3[ i ]
			}



			// ===>>> Begin process in multiple threads.

/*

			// Define the data for thread 0
			My_Data_0 :: struct {

			    use_threads : int,

				w1 : int,
				w2 : int,
				w3 : int,

				dW1 : [ ]f32,
				dW2 : [ ]f32,
				dW3 : [ ]f32,

				tcs : [ ]Thread_Ctx
			}

			// Fill the data.
			my_data_0 := My_Data_0 {

                use_threads = use_threads,

                w1 = w1,
                w2 = w2,
                w3 = w3,

                dW1 = dW1,
                dW2 = dW2,
                dW3 = dW3,

                tcs = tcs
			}

			thread_0 :: proc "c" ( arg : rawptr ) ->
			                       rawptr {

				// context = runtime.default_context( )

				// Get the data.
				data :=  ( ^My_Data_0 ) ( arg )

				use_threads := data^.use_threads

				w1 := data^.w1
                w2 := data^.w2
                w3 := data^.w3

                dW1 := data^.dW1
                dW2 := data^.dW2
                dW3 := data^.dW3

                tcs := data^.tcs

    			for t in 0 ..< use_threads {

    				for i in 0 ..< w1 do dW1[ i ] += tcs[ t ].dW1[ i ]
    				for i in 0 ..< w2 do dW2[ i ] += tcs[ t ].dW2[ i ]
    				for i in 0 ..< w3 do dW3[ i ] += tcs[ t ].dW3[ i ]
    			}

                return nil
			}


			// Define the data for thread 1
			My_Data_1 :: struct {

			    use_threads : int,
				net         : Net,
				db1         : [ ]f32,
				db2         : [ ]f32,
				db3         : [ ]f32,
				tcs         : [ ]Thread_Ctx
			}

			// Fill the data.
			my_data_1 := My_Data_1 {

                use_threads = use_threads,
                net         = net,
                db1         = db1,
                db2         = db2,
                db3         = db3,
                tcs         = tcs
			}

			thread_1 :: proc "c" ( arg : rawptr ) ->
			                      rawptr {

				// context = runtime.default_context( )

				// Get the data.
				data :=  ( ^My_Data_1 ) ( arg )

				use_threads := data^.use_threads

				net := data^.net

                db1 := data^.db1
                db2 := data^.db2
                db3 := data^.db3

                tcs := data^.tcs

    			for t in 0 ..< use_threads {

    				for i in 0 ..< net.l1.out_dim do db1[ i ] += tcs[ t ].db1[ i ]
    				for i in 0 ..< net.l2.out_dim do db2[ i ] += tcs[ t ].db2[ i ]
    				for i in 0 ..< net.l3.out_dim do db3[ i ] += tcs[ t ].db3[ i ]
    			}

                return nil
            }

			my_use_threads := 2

			my_ths  := make( [ ]posix.pthread_t, my_use_threads )
			my_rc_0 : posix.Errno = posix.pthread_create( & my_ths[ 0 ], nil, thread_0, rawptr( & my_data_0 ) )
			if my_rc_0 != .NONE do die( "ERROR : pthread_create failed." )
			my_rc_1 : posix.Errno = posix.pthread_create( & my_ths[ 1 ], nil, thread_1, rawptr( & my_data_1 ) )
			if my_rc_1 != .NONE do die( "ERROR : pthread_create failed." )

			/*
			for t in 0 ..< use_threads {

				for i in 0 ..< w1             do dW1[ i ] += tcs[ t ].dW1[ i ]
			    for i in 0 ..< w2             do dW2[ i ] += tcs[ t ].dW2[ i ]
				for i in 0 ..< w3             do dW3[ i ] += tcs[ t ].dW3[ i ]
				for i in 0 ..< net.l1.out_dim do db1[ i ] += tcs[ t ].db1[ i ]
				for i in 0 ..< net.l2.out_dim do db2[ i ] += tcs[ t ].db2[ i ]
				for i in 0 ..< net.l3.out_dim do db3[ i ] += tcs[ t ].db3[ i ]
			}
			*/


    		for t in 0 ..< my_use_threads {

                _ = posix.pthread_join( my_ths[ t ], nil )
    		}

            delete( my_ths )

*/

            // ===> End process in multiple threads.


			inv_bs : f32 = 1.0 / f32( bs )

			invert( w1, dW1, inv_bs )
			invert( w2, dW1, inv_bs )
			invert( w3, dW1, inv_bs )

			invert( net.l1.out_dim, db1, inv_bs )
			invert( net.l2.out_dim, db2, inv_bs )
			invert( net.l3.out_dim, db3, inv_bs )

			net.step += 1

			// Update the learning rate ( Goes backwords from the T_max to t_min ).

			t : f32 = 1.0 - 1.0 / ( f32( args.epochs) / f32( epoch ) )

			// args.lr = args.lr_start - t * ( args.lr_stop - args.lr_start )

			lr_min := args.lr_start
			lr_max := args.lr_stop
			args.lr = log_lerp_lr( lr_min, lr_max, t )

			adam_update( net.l1.W, net.l1.mW, net.l1.vW, dW1, args.lr, beta1, beta2, eps, net.step )
			adam_update( net.l2.W, net.l2.mW, net.l2.vW, dW2, args.lr, beta1, beta2, eps, net.step )
			adam_update( net.l3.W, net.l3.mW, net.l3.vW, dW3, args.lr, beta1, beta2, eps, net.step )

			adam_update( net.l1.b, net.l1.mb, net.l1.vb, db1, args.lr, beta1, beta2, eps, net.step )
			adam_update( net.l2.b, net.l2.mb, net.l2.vb, db2, args.lr, beta1, beta2, eps, net.step )
			adam_update( net.l3.b, net.l3.mb, net.l3.vb, db3, args.lr, beta1, beta2, eps, net.step )

			epoch_loss += batch_loss
			seen       += bs
			start      += args.batch
		}

		train_loss := f32( epoch_loss / f64( seen ) )
		acc := evaluate_accuracy_mt( & net, & test, ntest, nthreads, tcs[ : ] )
		fmt.printfln( "Epoch %d / %d  loss=%.5f  test_acc=%.4f  learning_rate=%0.6f",
	                  epoch, args.epochs, train_loss, acc, args.lr )
	}

	net_save( & net, args.weights_save )
	fmt.printfln( "Saved weights to %s", args.weights_save )

	for t in 0 ..< nthreads {

	    thread_ctx_free( & tcs[ t ] )
	}

	delete( tcs )

	delete( dW1 )
    delete( dW2 )
    delete( dW3 )

	delete( db1 )
	delete( db2 )
	delete( db3 )

	delete( perm )

	net_free( & net )
	mnist_free( & train )
	mnist_free( & test )
}

//
//  1.58 bits 5x times compression against  a 1 byte  i8  Weight
//  1.58 bits 20x times compression against a 4 bytes f32 Weight
//
//   The "secret" ternary encoding not binary.
//

test_ternary_encoding_decoding :: proc ( ) {

    data_slice_i8_start : [ 5 ]i8 = { -1, 0, 1, -1, 1 }

    data_slice_u8_in    : [ 5 ]u8

    // 1. Ternary enconding

    val_u8 : u8

    // 2. Ternary decoding

    data_slice_u8_out   : [ 5 ]u8

    data_slice_i8_end   : [ 5 ]i8

    // Start encoding.
    //
    from_1_58_to_ternary( data_slice_i8_start[ : ], data_slice_u8_in[ : ] )
    val_u8 = ternary_to_decimal( trits = data_slice_u8_in[ : ], num_trits = 5 )

    // Start decoding
    //
    decimal_to_ternary( n=val_u8, max_trits = 5, trits=data_slice_u8_out[ : ] )
    from_ternary_to_1_58( data_slice_u8_out[ : ], data_slice_i8_end[ : ] )

    // Comparing...

    flag_error := false
    for i in 0 ..< 5 {

        if data_slice_i8_start[ i ] != data_slice_i8_end[ i ] {

            flag_error = true

            fmt.printfln( "data_slice_i8_start : %v", data_slice_i8_start )
            fmt.printfln( "data_slice_i8_end   : %v", data_slice_i8_end )

            fmt.printfln( "TEST ERROR : In teste of 1.58 bits in i8 to a ternary u8 and back again to 1.58 bits." )
            os.exit( -1 )
        }
    }

    fmt.printfln( "data_slice_i8_start : %v", data_slice_i8_start )
    fmt.printfln( "data_slice_i8_end   : %v", data_slice_i8_end )

    fmt.printfln( "TEST SUCCESS: In teste of 1.58 bits in i8 to a ternary u8 and back again to 1.58 bits." )
    os.exit( -1 )
}

// input  -1, 0, 1      datatype i8
//         |  |  |
// output  2, 0, 1      datatype u8
from_1_58_to_ternary :: #force_inline proc "contextless" (
                                            in_slice_i8_vals  : [ ]i8,
                                            out_slice_u8_vals : [ ]u8 ) {

    for elem, i in in_slice_i8_vals{

        out_slice_u8_vals[ i ] = u8( in_slice_i8_vals[ i ] )
        if in_slice_i8_vals[ i ] == -1 {

            out_slice_u8_vals[ i ] = 2
        }
    }
}

// input   2, 0, 1      datatype u8
//         |  |  |
// output -1, 0, 1      datatype i8
from_ternary_to_1_58 :: #force_inline proc "contextless" (
                                            in_slice_u8_vals  : [ ]u8,
                                            out_slice_i8_vals : [ ]i8 ) {

    for elem, i in in_slice_u8_vals {

        out_slice_i8_vals[ i ] = i8( elem )
        if in_slice_u8_vals[ i ] == 2 {

            out_slice_i8_vals[ i ] = -1
        }
    }
}

decimal_to_ternary :: proc ( n         : u8,
                             max_trits : int,
                             trits     : [ ]u8 ) {

    n := n

    i := 0

    for n > 0 && i < max_trits {

        trits[ i ] = n % 3
        n /= 3
        i += 1
    }

    // Preenche o resto com zeros
    for i < max_trits {

        trits[ i ] = 0
        i += 1
    }

}

ternary_to_decimal :: proc ( trits     : [ ]u8,
                             num_trits : int ) ->
                             u8 {

    n : u8 = 0
    power : u8 = 1

    for i in 0 ..< num_trits {

        n += trits[ i ] * power;
        power *= 3
    }

    return n
}
