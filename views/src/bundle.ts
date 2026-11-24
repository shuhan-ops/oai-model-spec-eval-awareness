// The bundle's entry: what this project contributes to the viewer. `fxtr view` builds it into
// dist/ and serves it, rebuilding it as it changes (`fxtr views build` builds it without the
// viewer); each renderer is for a slot, an entity type (or a step or workflow's name), and has an
// id unique in the viewer.

import type { ViewBundle } from "fxtr-view";

const bundle: ViewBundle = {
  renderers: [
  ],
};

export default bundle;
