// Gate 4 fixed-mode argument safety: no arbitrary runner or child options.
    check({"rad50-tester","--test","dmabuf-live","--device","0",
           "--iterations","10","--rows","1","--lease-confirmed"},
          true,"live one-row smoke");
    check({"rad50-tester","--test","dmabuf-live","--device","1",
           "--iterations","100","--rows","768","--lease-confirmed"},
          true,"live 768 rows repeated generations");
    check({"rad50-tester","--test","dmabuf-live","--device","1",
           "--iterations","10","--rows","16"},
          false,"live requires exclusive lease acknowledgment");
    check({"rad50-tester","--test","dmabuf-live","--device","0",
           "--iterations","10","--lease-confirmed"},
          false,"live requires explicit bounded row count");
    check({"rad50-tester","--test","dmabuf-live","--device","0",
           "--rows","16","--lease-confirmed"},
          false,"live requires bounded iteration count");
    check({"rad50-tester","--test","dmabuf-live","--device","0",
           "--iterations","10","--rows","2","--lease-confirmed"},
          false,"live rejects unsupported row count");
    check({"rad50-tester","--test","dmabuf-live","--device","0",
           "--iterations","10","--rows","16","--mode","direct","--lease-confirmed"},
          false,"live rejects non-live CPU method flag");
    check({"rad50-tester","--test","dmabuf-live","--device","0",
           "--iterations","10","--rows","16","--rows","64","--lease-confirmed"},
          false,"live rejects duplicate rows selector");
    check({"rad50-tester","--test","dmabuf-live","--device","0",
           "--iterations","201","--rows","16","--lease-confirmed"},
          false,"live rejects unbounded iterations");
    check({"rad50-tester","--test","dmabuf-live-parallel","--device","0",
           "--iterations","10","--rows","768","--lease-confirmed"},
          true,"direct child publication plus parallel verifier");
    check({"rad50-tester","--test","dmabuf-live-parallel","--device","1",
           "--iterations","100","--rows","256","--lease-confirmed"},
          true,"bounded repeated direct parallel consumer");
    check({"rad50-tester","--test","dmabuf-live-parallel","--device","0",
           "--iterations","10","--rows","768"},
          false,"direct parallel consumer requires lease acknowledgement");
    check({"rad50-tester","--test","dmabuf-live-parallel","--device","0",
           "--iterations","10","--rows","999","--lease-confirmed"},
          false,"direct parallel consumer rejects unsupported rows");
    check({"rad50-tester","--test","dmabuf-live-relay","--device","0",
           "--iterations","10","--rows","1","--lease-confirmed"},
          true,"coherent parent relay single-row");
    check({"rad50-tester","--test","dmabuf-live-relay","--device","1",
           "--iterations","100","--rows","768","--lease-confirmed"},
          true,"coherent parent relay large-row");
    check({"rad50-tester","--test","dmabuf-live-relay","--device","0",
           "--iterations","10","--rows","768"},
          false,"coherent parent relay requires GPU lease acknowledgement");
    check({"rad50-tester","--test","dmabuf-live-relay","--device","0",
           "--iterations","10","--rows","2","--lease-confirmed"},
          false,"coherent parent relay rejects unsupported rows");
    check({"rad50-tester","--test","dmabuf-live-fg","--device","0",
           "--iterations","10","--rows","768","--lease-confirmed"},
          true,"fine-grained live XTX candidate");
    check({"rad50-tester","--test","dmabuf-live-fg","--device","1",
           "--iterations","100","--rows","16","--lease-confirmed"},
          true,"fine-grained live repeated generations");
    check({"rad50-tester","--test","dmabuf-live-fg","--device","0",
           "--iterations","10","--rows","768"},
          false,"fine-grained live requires lease");
    check({"rad50-tester","--test","dmabuf-live-fg","--device","0",
           "--iterations","10","--rows","1000","--lease-confirmed"},
          false,"fine-grained live bounded rows");

    // S4: fixed GPU verifier grid sweep; only parallel and relay can use it.
    check({"rad50-tester","--test","dmabuf-live-parallel","--device","0",
           "--iterations","10","--rows","768","--verifier-blocks","1",
           "--lease-confirmed"},true,"one-block full-bit verifier isolation");
    check({"rad50-tester","--test","dmabuf-live-relay","--device","1",
           "--iterations","100","--rows","768","--verifier-blocks","128",
           "--lease-confirmed"},true,"128-block coherent relay");
    check({"rad50-tester","--test","dmabuf-live-parallel","--device","0",
           "--iterations","10","--rows","768","--verifier-blocks","2",
           "--lease-confirmed"},false,"no arbitrary verifier grid");
    check({"rad50-tester","--test","dmabuf-live-parallel","--device","0",
           "--iterations","10","--rows","768","--verifier-blocks","32",
           "--verifier-blocks","64","--lease-confirmed"},false,"no repeated verifier grids");
    check({"rad50-tester","--test","dmabuf-live","--device","0",
           "--iterations","10","--rows","768","--verifier-blocks","32",
           "--lease-confirmed"},false,"not valid for legacy one-block reader");
    check({"rad50-tester","--test","combine-ab","--device","0",
           "--iterations","10","--mode","direct","--rows","768",
           "--verifier-blocks","32","--lease-confirmed"},false,"not valid for serial combine");
    check({"rad50-tester","--test","dmabuf-live-parallel","--device","0",
           "--iterations","10","--rows","768","--verifier-blocks","32"},
          false,"cannot bypass GPU lease acknowledgement");
