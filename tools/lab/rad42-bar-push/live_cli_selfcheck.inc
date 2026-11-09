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
