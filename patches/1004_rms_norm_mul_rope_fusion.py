1	"""Upstream backport: fuse rms_norm + mul + rope (+ view + set_rows).
2	
3	Cherry-picked from upstream commit 687e7789271ec1276e3470f158428e11a4f80b6f
4	(merged to ggml-org/llama.cpp master 2026-08-08, PR #26767 -- already merged
5	and upstream-CI-tested, not an open/unreviewed proposal).
6	
7	Fuses a chain of per-layer ops that runs every transformer layer, every
8	token -- rms_norm, mul (scale), rope (rotary position embedding), and
9	optionally the following view/set_rows -- into a single kernel launch,
10	instead of the four-to-five separate launches upstream's unfused graph
11	issues today. Not gated to CUDA-only (no ``GGML_USE_HIP`` guard anywhere in
12	the diff), applies to HIP builds as-is.
13	
14	Same category as the already-validated RDNA4 Q6_K/Q2_K MMQ fix in spirit --
15	a change to what runs, not which candidate this project's autotune dispatch
16	engine picks, so the tuner cannot reach or discover this on its own. But
17	it's not a MUL_MAT-family change at all: this is the norm/rope pipeline,
18	entirely outside anything this session's HI44-56 dispatch-engine work
19	touches.
20	
21	All three files this backport touches were checked against the current
22	pinned base (22dc605) before writing anchors: the new code is purely
23	additive (new functions appended, new declaration appended, two new
24	`ggml_cuda_can_fuse`/`ggml_cuda_try_fuse` checks inserted ahead of existing
25	ones) and every helper it calls (`ggml_cuda_check_fusion_memory_ranges`,
26	`ggml_cuda_pdl_lc`/`_sync`, `fastmodulo`, `init_fastdiv_values`,
27	`block_reduce_method`, `rope_yarn`) already exists in this project's base
28	commit, confirmed by direct grep before any edit was written -- this is not
29	introducing a dependency our base predates.
30	
31	Checked for anchor conflicts against this project's own three existing
32	`ggml-cuda.cu` patches (`0200_dispatch_hook.py`, `0700_coverage_counters.py`,
33	`0900_pool_workspace_metrics.py`): none of their anchors fall anywhere near
34	`ggml_cuda_can_fuse`/`ggml_cuda_try_fuse` (~line 2650-3950) -- they touch
35	`ggml_backend_cuda_free`, `ggml_cuda_mul_mat_id`, and pool alloc/free
36	functions instead. No real overlap, despite all four patches sharing a file.
37	"""
38	
39	import re
40	
41	from bigcherry.patcher import Edit, FilePatch
42	
43	GROUP = "upstream-fixes"
44	
45	
46	def re_escape_literal(s: str) -> str:
47	    return re.escape(s)
48	
49	# ---------------------------------------------------------------- rope.cuh
50	
51	_ROPE_CUH_DECL = (
52	    "void ggml_cuda_op_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * set_rows);\n"
53	    "\n"
54	    "void ggml_cuda_op_rms_norm_mul_rope_fused(ggml_backend_cuda_context & ctx, "
55	    "ggml_tensor * rms_norm, ggml_tensor * mul, ggml_tensor * rope, ggml_tensor * set_rows);"
56	)
57	
58	ROPE_CUH_PATCH = FilePatch(
59	    path="ggml/src/ggml-cuda/rope.cuh",
60	    description="Declare ggml_cuda_op_rms_norm_mul_rope_fused (upstream PR #26767)",
61	    edits=(
62	        Edit(
63	            id="rope-cuh-decl",
64	            anchor=r"^void ggml_cuda_op_rope_fused\(ggml_backend_cuda_context & ctx, "
65	                   r"ggml_tensor \* dst, ggml_tensor \* set_rows\);$",
66	            rationale="the declaration of the existing rope+set_rows fusion entry point",
67	            mode="replace",
68	            text=_ROPE_CUH_DECL,
69	            guard=r"void ggml_cuda_op_rms_norm_mul_rope_fused\(",
70	        ),
71	    ),
72	)
73	
74	# ----------------------------------------------------------------- rope.cu
75	
76	# Verbatim from the upstream diff -- a purely new, self-contained kernel plus
77	# its two callers. Reproduced in full rather than reconstructed, since this
78	# is real numerical kernel code (block reduction, rope_yarn application,
79	# fastmod-based broadcast indexing for the mul operand) where a
80	# paraphrase risks a subtle transcription bug the compiler won't catch.
81	_ROPE_CU_BODY = '''
82	
83	// fused RMS_NORM + MUL + ROPE (+ VIEW + SET_ROWS)
84	// one block per row: block_reduce gives the norm scale, then each thread applies mul and rope to the elements it owns
85	template <int block_size, bool has_ff, typename D>
86	static __global__ void rms_norm_mul_rope_f32(
87	        const float * x, D * dst, const int ncols,
88	        const int64_t s01, const int64_t s02, const int64_t s03,
89	        const int64_t s1, const int64_t s2, const int64_t s3,
90	        const float eps,
91	        const float * mul,
92	        const int64_t mul_s01, const int64_t mul_s02, const int64_t mul_s03,
93	        const uint3 mul_ncols_packed, const uint3 mul_nrows_packed,
94	        const uint3 mul_nchannels_packed, const uint3 mul_nsamples_packed,
95	        const int n_dims, const int32_t * pos,
96	        const float freq_scale, const float ext_factor, const float attn_factor,
97	        const rope_corr_dims corr_dims, const float theta_scale,
98	        const float * freq_factors,
99	        const int64_t * row_indices, const int set_rows_stride,
100	        const bool is_neox) {
101	    ggml_cuda_pdl_lc();
102	    const int row     = blockIdx.x;
103	    const int channel = blockIdx.y;
104	    const int sample  = blockIdx.z;
105	    const int tid     = threadIdx.x;
106	
107	    x += sample*s03 + channel*s02 + row*s01;
108	
109	    const uint32_t mul_row     = fastmodulo(row,     mul_nrows_packed);
110	    const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
111	    const uint32_t mul_sample  = fastmodulo(sample,  mul_nsamples_packed);
112	    mul += mul_sample*mul_s03 + mul_channel*mul_s02 + mul_row*mul_s01;
113	
114	    float tmp = 0.0f;
115	
116	    ggml_cuda_pdl_sync();
117	    for (int col = tid; col < ncols; col += block_size) {
118	        const float xi = x[col];
119	        tmp += xi * xi;
120	    }
121	
122	    extern __shared__ float s_sum[];
123	    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
124	
125	    const float scale = rsqrtf(tmp/ncols + eps);
126	
127	    int64_t idst = sample*s3 + channel*s2 + row*s1;
128	    if (set_rows_stride != 0) {
129	        idst = row*s1 + row_indices[channel]*set_rows_stride;
130	    }
131	    dst += idst;
132	
133	    for (int i0 = 2*tid; i0 < ncols; i0 += 2*block_size) {
134	        int ix0;
135	        int ix1;
136	        if (is_neox && i0 < n_dims) {
137	            ix0 = i0/2;
138	            ix1 = i0/2 + n_dims/2;
139	        } else {
140	            ix0 = i0 + 0;
141	            ix1 = i0 + 1;
142	        }
143	
144	        const float x0 = scale * x[ix0] * mul[fastmodulo(ix0, mul_ncols_packed)];
145	        const float x1 = scale * x[ix1] * mul[fastmodulo(ix1, mul_ncols_packed)];
146	
147	        if (i0 >= n_dims) {
148	            dst[ix0] = ggml_cuda_cast<D>(x0);
149	            dst[ix1] = ggml_cuda_cast<D>(x1);
150	            continue;
151	        }
152	
153	        const float theta_base  = pos[channel]*powf(theta_scale, i0/2.0f);
154	        const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;
155	
156	        float cos_theta;
157	        float sin_theta;
158	        rope_yarn<true>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);
159	
160	        dst[ix0] = ggml_cuda_cast<D>(x0*cos_theta - x1*sin_theta);
161	        dst[ix1] = ggml_cuda_cast<D>(x0*sin_theta + x1*cos_theta);
162	    }
163	}
164	
165	template <typename D>
166	static void rms_norm_mul_rope_cuda(
167	        const float * x, D * dst,
168	        const int ncols, const int nrows, const int nchannels, const int nsamples,
169	        const int64_t s01, const int64_t s02, const int64_t s03,
170	        const int64_t s1, const int64_t s2, const int64_t s3,
171	        const float eps,
172	        const float * mul,
173	        const int64_t mul_s01, const int64_t mul_s02, const int64_t mul_s03,
174	        const uint32_t mul_ncols, const uint32_t mul_nrows,
175	        const uint32_t mul_nchannels, const uint32_t mul_nsamples,
176	        const int n_dims, const int32_t * pos,
177	        const float freq_scale, const float freq_base, const float ext_factor, const float attn_factor,
178	        const rope_corr_dims corr_dims,
179	        const float * freq_factors,
180	        const int64_t * row_indices, const int set_rows_stride,
181	        const bool is_neox, cudaStream_t stream) {
182	    GGML_ASSERT(ncols % 2 == 0);
183	
184	    const dim3 blocks_num(nrows, nchannels, nsamples);
185	
186	    const float theta_scale = powf(freq_base, -2.0f/n_dims);
187	
188	    const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
189	    const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
190	    const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
191	    const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);
192	
193	    if (ncols < 1024) {
194	        const dim3 block_dims(256, 1, 1);
195	        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, 32*sizeof(float), stream};
196	        if (freq_factors == nullptr) {
197	            ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<256, false, D>, launch_params,
198	                x, dst, ncols, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
199	                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
200	                n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
201	                freq_factors, row_indices, set_rows_stride, is_neox);
202	        } else {
203	            ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<256, true, D>, launch_params,
204	                x, dst, ncols, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
205	                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
206	                n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
207	                freq_factors, row_indices, set_rows_stride, is_neox);
208	        }
209	    } else {
210	        const dim3 block_dims(1024, 1, 1);
211	        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, 32*sizeof(float), stream};
212	        if (freq_factors == nullptr) {
213	            ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<1024, false, D>, launch_params,
214	                x, dst, ncols, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
215	                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
216	                n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
217	                freq_factors, row_indices, set_rows_stride, is_neox);
218	        } else {
219	            ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<1024, true, D>, launch_params,
220	                x, dst, ncols, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
221	                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
222	                n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
223	                freq_factors, row_indices, set_rows_stride, is_neox);
224	        }
225	    }
226	}
227	
228	void ggml_cuda_op_rms_norm_mul_rope_fused(ggml_backend_cuda_context & ctx,
229	        ggml_tensor * rms_norm, ggml_tensor * mul, ggml_tensor * rope, ggml_tensor * set_rows) {
230	    const ggml_tensor * x = rms_norm->src[0];
231	    const ggml_tensor * mul_src = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];
232	
233	    float eps = 0.0f;
234	    memcpy(&eps, rms_norm->op_params, sizeof(float));
235	    GGML_ASSERT(eps >= 0.0f);
236	
237	    GGML_ASSERT(x->type == GGML_TYPE_F32);
238	    GGML_ASSERT(mul_src->type == GGML_TYPE_F32);
239	    GGML_ASSERT(rope->type == GGML_TYPE_F32);
240	
241	    void *          dst_d           = rope->data;
242	    ggml_type       dst_type        = rope->type;
243	    const int64_t * row_indices     = nullptr;
244	    int             set_rows_stride = 0;
245	
246	    if (set_rows != nullptr) {
247	        dst_d           = set_rows->data;
248	        dst_type        = set_rows->type;
249	        row_indices     = (const int64_t *) set_rows->src[1]->data;
250	        set_rows_stride = set_rows->nb[1] / ggml_type_size(set_rows->type);
251	    }
252	
253	    const int n_dims     = ((const int32_t *) rope->op_params)[1];
254	    const int mode       = ((const int32_t *) rope->op_params)[2];
255	    const int n_ctx_orig = ((const int32_t *) rope->op_params)[4];
256	
257	    float freq_base;
258	    float freq_scale;
259	    float ext_factor;
260	    float attn_factor;
261	    float beta_fast;
262	    float beta_slow;
263	
264	    memcpy(&freq_base,   (const int32_t *) rope->op_params +  5, sizeof(float));
265	    memcpy(&freq_scale,  (const int32_t *) rope->op_params +  6, sizeof(float));
266	    memcpy(&ext_factor,  (const int32_t *) rope->op_params +  7, sizeof(float));
267	    memcpy(&attn_factor, (const int32_t *) rope->op_params +  8, sizeof(float));
268	    memcpy(&beta_fast,   (const int32_t *) rope->op_params +  9, sizeof(float));
269	    memcpy(&beta_slow,   (const int32_t *) rope->op_params + 10, sizeof(float));
270	
271	    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;
272	
273	    const int32_t * pos = (const int32_t *) rope->src[1]->data;
274	
275	    const float * freq_factors = rope->src[2] != nullptr ? (const float *) rope->src[2]->data : nullptr;
276	
277	    rope_corr_dims corr_dims;
278	    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims.v);
279	
280	    const size_t ts0 = ggml_type_size(x->type);
281	    GGML_ASSERT(x->nb[0] == ts0);
282	    const int64_t s01 = x->nb[1] / ts0;
283	    const int64_t s02 = x->nb[2] / ts0;
284	    const int64_t s03 = x->nb[3] / ts0;
285	
286	    const size_t ts_mul = ggml_type_size(mul_src->type);
287	    GGML_ASSERT(mul_src->nb[0] == ts_mul);
288	    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
289	    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
290	    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;
291	
292	    const size_t ts_dst = ggml_type_size(rope->type);
293	    const int64_t s1 = rope->nb[1] / ts_dst;
294	    const int64_t s2 = rope->nb[2] / ts_dst;
295	    const int64_t s3 = rope->nb[3] / ts_dst;
296	
297	    cudaStream_t stream = ctx.stream();
298	
299	    if (dst_type == GGML_TYPE_F32) {
300	        rms_norm_mul_rope_cuda((const float *) x->data, (float *) dst_d,
301	            x->ne[0], x->ne[1], x->ne[2], x->ne[3], s01, s02, s03, s1, s2, s3, eps,
302	            (const float *) mul_src->data, mul_s01, mul_s02, mul_s03,
303	            mul_src->ne[0], mul_src->ne[1], mul_src->ne[2], mul_src->ne[3],
304	            n_dims, pos, freq_scale, freq_base, ext_factor, attn_factor, corr_dims,
305	            freq_factors, row_indices, set_rows_stride, is_neox, stream);
306	    } else if (dst_type == GGML_TYPE_F16) {
307	        rms_norm_mul_rope_cuda((const float *) x->data, (half *) dst_d,
308	            x->ne[0], x->ne[1], x->ne[2], x->ne[3], s01, s02, s03, s1, s2, s3, eps,
309	            (const float *) mul_src->data, mul_s01, mul_s02, mul_s03,
310	            mul_src->ne[0], mul_src->ne[1], mul_src->ne[2], mul_src->ne[3],
311	            n_dims, pos, freq_scale, freq_base, ext_factor, attn_factor, corr_dims,
312	            freq_factors, row_indices, set_rows_stride, is_neox, stream);
313	    } else {
314	        GGML_ABORT("fatal error");
315	    }
316	}'''
317	
318	ROPE_CU_PATCH = FilePatch(
319	    path="ggml/src/ggml-cuda/rope.cu",
320	    description="Add the fused rms_norm+mul+rope(+set_rows) kernel and its "
321	                "entry point (upstream PR #26767)",
322	    edits=(
323	        Edit(
324	            id="rope-cu-fused-kernel",
325	            anchor=r"^void ggml_cuda_op_rope_fused\(ggml_backend_cuda_context & ctx, "
326	                   r"ggml_tensor \* rope, ggml_tensor \* set_rows\) \{\n"
327	                   r"    ggml_cuda_op_rope_impl<true>\(ctx, rope, set_rows\);\n"
328	                   r"\}",
329	            rationale="the last function in the file, ggml_cuda_op_rope_fused",
330	            mode="insert_after",
331	            text=_ROPE_CU_BODY,
332	            guard=r"void ggml_cuda_op_rms_norm_mul_rope_fused\(ggml_backend_cuda_context & ctx,",
333	        ),
334	    ),
335	)
336	
337	# ------------------------------------------------------------- ggml-cuda.cu
338	
339	_SHOULD_FUSE_FN = '''
340	
341	static bool ggml_cuda_should_fuse_rms_norm_mul_rope(const ggml_tensor * rms_norm,
342	                                                    const ggml_tensor * mul,
343	                                                    const ggml_tensor * rope) {
344	    if (rms_norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || rope->op != GGML_OP_ROPE) {
345	        return false;
346	    }
347	
348	    if (rms_norm->src[0]->type != GGML_TYPE_F32 || rms_norm->type != GGML_TYPE_F32 ||
349	        mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 ||
350	        mul->type != GGML_TYPE_F32 || rope->type != GGML_TYPE_F32) {
351	        return false;
352	    }
353	
354	    if (rope->src[0] != mul) {
355	        return false;
356	    }
357	
358	    //if rms norm is the B operand, then we don't handle broadcast
359	    if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
360	        return false;
361	    }
362	
363	    if (!ggml_are_same_shape(rms_norm, mul)) {
364	        return false;
365	    }
366	
367	    //rms_norm kernel assumes contiguous rows
368	    if (!ggml_is_contiguous_rows(rms_norm->src[0]) ||
369	        !ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
370	        return false;
371	    }
372	
373	    // the fused kernel handles the norm/neox rope modes only
374	    const int mode = ((const int32_t *) rope->op_params)[2];
375	    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
376	        return false;
377	    }
378	
379	    const int n_dims = ((const int32_t *) rope->op_params)[1];
380	    if (n_dims % 2 != 0 || rope->src[0]->ne[0] % 2 != 0) {
381	        return false;
382	    }
383	
384	    return true;
385	}'''
386	
387	_CAN_FUSE_BLOCK_OLD = '''    std::initializer_list<enum ggml_op> rope_set_rows_ops = { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };
388	
389	    if (is_equal(rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
390	        const ggml_tensor * rope     = cgraph->nodes[node_idx];
391	        const ggml_tensor * view     = cgraph->nodes[node_idx + 1];
392	        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 2];
393	
394	        if (ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
395	            return true;
396	        }
397	    }'''
398	
399	_CAN_FUSE_BLOCK_NEW = '''    std::initializer_list<enum ggml_op> rms_norm_mul_rope_ops          = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE };
400	    std::initializer_list<enum ggml_op> rms_norm_mul_rope_set_rows_ops = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };
401	
402	    if (is_equal(rms_norm_mul_rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
403	        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
404	        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
405	        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];
406	        const ggml_tensor * view     = cgraph->nodes[node_idx + 3];
407	        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 4];
408	
409	        if (ggml_check_edges(cgraph, node_idx, {{1, 0, 0}, {2, 0, 1}, {3, 0, 2}, {4, 0, 3}}) &&
410	            ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope) &&
411	            ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
412	            int out_nodes[] = { node_idx + 4 };
413	            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
414	        }
415	    }
416	
417	    if (is_equal(rms_norm_mul_rope_ops, ops) && ggml_can_fuse(cgraph, node_idx, ops)) {
418	        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
419	        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
420	        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];
421	
422	        if (ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope)) {
423	            int out_nodes[] = { node_idx + 2 };
424	            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
425	        }
426	        return false;
427	    }
428	
429	    std::initializer_list<enum ggml_op> rope_set_rows_ops = { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };
430	
431	    if (is_equal(rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
432	        const ggml_tensor * rope     = cgraph->nodes[node_idx];
433	        const ggml_tensor * view     = cgraph->nodes[node_idx + 1];
434	        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 2];
435	
436	        if (ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
437	            int out_nodes[] = { node_idx + 2 };
438	            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
439	        }
440	    }'''
441	
442	_TRY_FUSE_OLD = "    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {"
443	
444	_TRY_FUSE_NEW = '''    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
445	        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 4]);
446	        return 4;
447	    }
448	
449	    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }, {})) {
450	        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], nullptr);
451	        return 2;
452	    }
453	
454	    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {'''
455	
456	GGML_CUDA_PATCH = FilePatch(
457	    path="ggml/src/ggml-cuda/ggml-cuda.cu",
458	    description="Wire the rms_norm+mul+rope fusion into ggml_cuda_can_fuse "
459	                "and ggml_cuda_try_fuse (upstream PR #26767)",
460	    edits=(
461	        Edit(
462	            id="ggml-cuda-should-fuse-fn",
463	            # The trailing comment (`// match gated_delta_net`) is blanked in
464	            # the noise-stripped copy the patcher matches against, so it can't
465	            # be part of the anchor even though it stays in the file
466	            # unchanged. Instead span from the function's own (unique)
467	            # signature, non-greedily, to its first `return true;` + closing
468	            # brace -- which is its own end, since the body only contains
469	            # `return false;` before that point -- and insert_after so the
470	            # body never has to be restated.
471	            anchor=r"^static bool ggml_cuda_should_fuse_rope_set_rows\(const ggml_tensor \* rope,\n"
472	                   r"[\s\S]*?"
473	                   r"    return true;\n\}",
474	            rationale="the whole of ggml_cuda_should_fuse_rope_set_rows, "
475	                      "ending at its own closing brace",
476	            mode="insert_after",
477	            text=_SHOULD_FUSE_FN,
478	            guard=r"ggml_cuda_should_fuse_rms_norm_mul_rope\(const ggml_tensor \* rms_norm,",
479	            max_span_lines=40,
480	        ),
481	        Edit(
482	            id="ggml-cuda-can-fuse-block",
483	            anchor=r"    std::initializer_list<enum ggml_op> rope_set_rows_ops = "
484	                   r"\{ GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS \};\n"
485	                   r"\n"
486	                   r"    if \(is_equal\(rope_set_rows_ops, ops\) && "
487	                   r"ggml_can_fuse_subgraph\(cgraph, node_idx, ops, \{ node_idx \+ 2 \}\)\) \{\n"
488	                   r"        const ggml_tensor \* rope     = cgraph->nodes\[node_idx\];\n"
489	                   r"        const ggml_tensor \* view     = cgraph->nodes\[node_idx \+ 1\];\n"
490	                   r"        const ggml_tensor \* set_rows = cgraph->nodes\[node_idx \+ 2\];\n"
491	                   r"\n"
492	                   r"        if \(ggml_cuda_should_fuse_rope_set_rows\(rope, view, set_rows\)\) \{\n"
493	                   r"            return true;\n"
494	                   r"        \}\n"
495	                   r"    \}",
496	            rationale="the rope+view+set_rows fusion check inside "
497	                      "ggml_cuda_can_fuse, which the new rms_norm+mul+rope "
498	                      "checks are inserted ahead of, plus its own return "
499	                      "true replaced with the memory-range-checked form",
500	            mode="replace",
501	            text=_CAN_FUSE_BLOCK_NEW,
502	            guard=r"rms_norm_mul_rope_set_rows_ops = \{ GGML_OP_RMS_NORM, GGML_OP_MUL, "
503	                  r"GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS \};",
504	            max_span_lines=15,
505	        ),
506	        Edit(
507	            id="ggml-cuda-try-fuse-block",
508	            anchor=re_escape_literal(_TRY_FUSE_OLD),
509	            rationale="the RMS_NORM+MUL+ADD fusion attempt inside "
510	                      "ggml_cuda_try_fuse, ahead of which the two new "
511	                      "rms_norm+mul+rope attempts are inserted",
512	            mode="replace",
513	            text=_TRY_FUSE_NEW,
514	            guard=r"ggml_cuda_op_rms_norm_mul_rope_fused\(\*cuda_ctx, node, "
515	                  r"cgraph->nodes\[i \+ 1\], cgraph->nodes\[i \+ 2\], cgraph->nodes\[i \+ 4\]\);",
516	        ),
517	    ),
518	)
519	
520	
521	PATCHES = [ROPE_CUH_PATCH, ROPE_CU_PATCH, GGML_CUDA_PATCH]
522	