.. raw:: html

   <link rel="preconnect" href="https://fonts.googleapis.com">
   <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
   <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&amp;family=Inter:wght@400;500;600&amp;family=JetBrains+Mono:wght@400;500&amp;display=swap" rel="stylesheet">

   <div class="hg-page">

     <section class="hg-hero">
       <div class="hg-inner hg-hero-grid">
         <div>
           <p class="hg-eyebrow">Python &middot; hyperspectral imaging</p>
           <h1>HyperGas</h1>
           <p class="hg-tagline">Trace gases, read from light itself.</p>
           <p class="hg-desc">
             HyperGas reads, processes, and writes data from hyperspectral imagers,
             turning L1 radiance data into trace-gas concentration enhancements and emission-rate estimates.
           </p>
           <div class="hg-btn-row">
             <a class="hg-btn hg-btn-primary" href="getting_started">
               Get started
               <svg width="15" height="15" viewBox="0 0 15 15" fill="none"><path d="M3 7.5H12M12 7.5L8 3.5M12 7.5L8 11.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
             </a>
             <a class="hg-btn hg-btn-ghost" href="https://github.com/SRON-ESG/HyperGas/">
               <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"/></svg>
               View on GitHub
             </a>
           </div>
           <div class="hg-chips">
             <span class="hg-chip">conda install -c conda-forge hypergas</span>
             <span class="hg-chip">EMIT &middot; EnMAP &middot; PRISMA</span>
           </div>
         </div>
         <div class="hg-plume-panel">
           <img src="_static/images/plume_art.png" alt="Rainbow-colored trace-gas plume flowing upward from a small source">
         </div>
       </div>
     </section>

     <section id="instruments">
       <div class="hg-inner">
         <div class="hg-section-head">
           <p class="hg-eyebrow" style="justify-content:center;">Supported instruments</p>
           <h2>One interface, every sensor</h2>
           <p>Reading is built on Satpy's hyperspectral readers, so new instruments plug in as they gain support.</p>
         </div>
         <div class="hg-instrument-grid">
           <div class="hg-instrument-card">
             <h3>EMIT</h3>
             <p>Earth Surface Mineral Dust Source Investigation, JPL / NASA.</p>
             <span class="hg-reader-badge">emit_l1b</span>
           </div>
           <div class="hg-instrument-card">
             <h3>EnMAP</h3>
             <p>Environmental Mapping and Analysis Program, DLR, GFZ.</p>
             <span class="hg-reader-badge">hsi_l1b</span>
           </div>
           <div class="hg-instrument-card">
             <h3>PRISMA</h3>
             <p>PRecursore IperSpettrale della Missione Applicativa, ASI.</p>
             <span class="hg-reader-badge">hyc_l1</span>
           </div>
         </div>
       </div>
     </section>

     <section class="hg-pipeline" id="pipeline">
       <div class="hg-inner">
         <div class="hg-section-head">
           <p class="hg-eyebrow" style="justify-content:center;">Processing levels</p>
           <h2>From radiance to emission rate</h2>
           <p>Each HyperGas run: radiance in, quantified emission rates out.</p>
         </div>
         <div class="hg-stepper">
           <span class="hg-step hg-step-l1">L1 &middot; radiance</span>
           <span class="hg-step-arrow"></span>
           <span class="hg-step hg-step-l2">L2 &middot; denoised &Delta;X</span>
           <span class="hg-step-arrow"></span>
           <span class="hg-step hg-step-l34">L3/L4 &middot; plume mask + emission estimation</span>
         </div>
         <div class="hg-pipeline-figure">
           <img src="_static/images/hypergas_pipeline.png" alt="HyperGas processing pipeline from L1 radiance through denoised trace-gas enhancement to L3/L4 plume mask and emission rate estimation">
         </div>
         <p class="hg-pipeline-caption">L1 radiance &rarr; L2 denoised trace-gas enhancement (&Delta;X) and automatic plume mask &rarr; L3/L4 emission estimation using the integrated mass enhancement (IME) and cross-sectional flux (CSF) methods.</p>
       </div>
     </section>

     <section id="features">
       <div class="hg-inner">
         <div class="hg-section-head">
           <p class="hg-eyebrow" style="justify-content:center;">Key features</p>
           <h2>What HyperGas does for you</h2>
         </div>
         <div class="hg-feature-grid">
           <div class="hg-feature-card" style="--card-accent:#4c8ef7;">
             <h3>RGB compositing</h3>
             <p>Combine multiple spectral bands into RGB images.</p>
           </div>
           <div class="hg-feature-card" style="--card-accent:#22c17f;">
             <h3>Trace gas retrieval</h3>
             <p>Retrieve enhancements for methane, carbon dioxide, and other trace gases.</p>
           </div>
           <div class="hg-feature-card" style="--card-accent:#f2b23e;">
             <h3>Denoising</h3>
             <p>Clean retrieval outputs to isolate real plume signal from noise.</p>
           </div>
           <div class="hg-feature-card" style="--card-accent:#ef8f4c;">
             <h3>Flexible export</h3>
             <p>Save results as PNG, HTML, or CF-compliant NetCDF for downstream tools.</p>
           </div>
           <div class="hg-feature-card" style="--card-accent:#ef6a4c;">
             <h3>Plume detection</h3>
             <p>Semi-supervised detection of gas plumes with automatic plume mask generation.</p>
           </div>
           <div class="hg-feature-card" style="--card-accent:#93a0b8;">
             <h3>Emission estimation</h3>
             <p>Estimate gas emission rates and write them out to CSV.</p>
           </div>
         </div>
       </div>
     </section>

     <section id="get-started">
       <div class="hg-inner hg-footer-grid">
         <div>
           <h3>Get started</h3>
           <p>Install HyperGas and read your first L1 radiance in a few lines.</p>
           <div class="hg-code-block">conda install -c conda-forge hypergas</div>
           <p style="margin-top:1rem;">See the <a href="getting_started" style="color:#4c8ef7;">quickstart guide</a> and <a href="api/modules.html" style="color:#4c8ef7;">API reference</a> for details.</p>
         </div>
         <div>
           <h3>Get involved</h3>
           <p>HyperGas is designed to make it straightforward to add trace-gas retrieval support for new HSI instruments.</p>
           <p>Contributions and issue reports are welcome on the <a href="https://github.com/SRON-ESG/HyperGas/" style="color:#4c8ef7;">GitHub repository</a>.</p>
         </div>
       </div>
     </section>

     <section class="hg-citation" id="citation">
       <div class="hg-inner">
         <div class="hg-section-head">
           <p class="hg-eyebrow" style="justify-content:center;">Research</p>
           <h2>Citation</h2>
           <p>If HyperGas contributes to your research, please cite the software and the associated publication. And don't forget to add your paper to the <a href="userguide/publication" style="color:#4c8ef7;">Publications page</a>.</p>
         </div>
         <div class="hg-citation-card">
           <div class="hg-citation-meta">
             <span class="hg-citation-label">Software DOI</span>
             <a class="hg-citation-doi" href="https://doi.org/10.5281/zenodo.18154956">10.5281/zenodo.18154956</a>
           </div>
           <div class="hg-citation-copy">
             <h3>Recommended citation</h3>
             <p>Zhang, X., Maasakkers, J. D., de Jong, T. A., Tol, P., Reuland, F., Brandt, A. R., Kort, E. A., Adams, T. J., and Aben, I. (2026). <em>HyperGas 1.0: a python package for analyzing hyperspectral data for greenhouse gases from retrieval to emission rate quantification, </em> Geosci. Model Dev., 19, 5979–6000, <a href=" https://doi.org/10.5194/gmd-19-5979-2026"> https://doi.org/10.5194/gmd-19-5979-2026.</a></p>
           </div>
         </div>
       </div>
     </section>

   </div>


.. title:: HyperGas's Documentation

.. toctree::
   :maxdepth: 2
   :hidden:

   Getting Started<getting_started/index>
   User Guide<userguide/index>
   Workflow<workflow/index>
   Developer Guide<developer_guide/index>
   API <api/modules>

