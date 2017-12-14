<!DOCTYPE html>
<html>
<head><meta charset="utf-8" />
<title>Stock-trend-prediction</title>

<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/2.0.3/jquery.min.js"></script>

<style type="text/css">
    /*!
*
* Twitter Bootstrap
*
*/
/*!
 * Bootstrap v3.3.6 (http://getbootstrap.com)
 * Copyright 2011-2015 Twitter, Inc.
 * Licensed under MIT (https://github.com/twbs/bootstrap/blob/master/LICENSE)
 */
/*! normalize.css v3.0.3 | MIT License | github.com/necolas/normalize.css */
html {
  font-family: sans-serif;
  -ms-text-size-adjust: 100%;
  -webkit-text-size-adjust: 100%;
}
body {
  margin: 0;
}
article,
aside,
details,
figcaption,
figure,
footer,
header,
hgroup,
main,
menu,
nav,
section,
summary {
  display: block;
}
audio,
canvas,
progress,
video {
  display: inline-block;
  vertical-align: baseline;
}
audio:not([controls]) {
  display: none;
  height: 0;
}
[hidden],
template {
  display: none;
}
a {
  background-color: transparent;
}
a:active,
a:hover {
  outline: 0;
}
abbr[title] {
  border-bottom: 1px dotted;
}
b,
strong {
  font-weight: bold;
}
dfn {
  font-style: italic;
}
h1 {
  font-size: 2em;
  margin: 0.67em 0;
}
mark {
  background: #ff0;
  color: #000;
}
small {
  font-size: 80%;
}
sub,
sup {
  font-size: 75%;
  line-height: 0;
  position: relative;
  vertical-align: baseline;
}
sup {
  top: -0.5em;
}
sub {
  bottom: -0.25em;
}
img {
  border: 0;
}
svg:not(:root) {
  overflow: hidden;
}
figure {
  margin: 1em 40px;
}
hr {
  box-sizing: content-box;
  height: 0;
}
pre {
  overflow: auto;
}
code,
kbd,
pre,
samp {
  font-family: monospace, monospace;
  font-size: 1em;
}
button,
input,
optgroup,
select,
textarea {
  color: inherit;
  font: inherit;
  margin: 0;
}
button {
  overflow: visible;
}
button,
select {
  text-transform: none;
}
button,
html input[type="button"],
input[type="reset"],
input[type="submit"] {
  -webkit-appearance: button;
  cursor: pointer;
}
button[disabled],
html input[disabled] {
  cursor: default;
}
button::-moz-focus-inner,
input::-moz-focus-inner {
  border: 0;
  padding: 0;
}
input {
  line-height: normal;
}
input[type="checkbox"],
input[type="radio"] {
  box-sizing: border-box;
  padding: 0;
}
input[type="number"]::-webkit-inner-spin-button,
input[type="number"]::-webkit-outer-spin-button {
  height: auto;
}
input[type="search"] {
  -webkit-appearance: textfield;
  box-sizing: content-box;
}
input[type="search"]::-webkit-search-cancel-button,
input[type="search"]::-webkit-search-decoration {
  -webkit-appearance: none;
}
fieldset {
  border: 1px solid #c0c0c0;
  margin: 0 2px;
  padding: 0.35em 0.625em 0.75em;
}
legend {
  border: 0;
  padding: 0;
}
textarea {
  overflow: auto;
}
optgroup {
  font-weight: bold;
}
table {
  border-collapse: collapse;
  border-spacing: 0;
}
td,
th {
  padding: 0;
}
/*! Source: https://github.com/h5bp/html5-boilerplate/blob/master/src/css/main.css */
@media print {
  *,
  *:before,
  *:after {
    background: transparent !important;
    color: #000 !important;
    box-shadow: none !important;
    text-shadow: none !important;
  }
  a,
  a:visited {
    text-decoration: underline;
  }
  a[href]:after {
    content: " (" attr(href) ")";
  }
  abbr[title]:after {
    content: " (" attr(title) ")";
  }
  a[href^="#"]:after,
  a[href^="javascript:"]:after {
    content: "";
  }
  pre,
  blockquote {
    border: 1px solid #999;
    page-break-inside: avoid;
  }
  thead {
    display: table-header-group;
  }
  tr,
  img {
    page-break-inside: avoid;
  }
  img {
    max-width: 100% !important;
  }
  p,
  h2,
  h3 {
    orphans: 3;
    widows: 3;
  }
  h2,
  h3 {
    page-break-after: avoid;
  }
  .navbar {
    display: none;
  }
  .btn > .caret,
  .dropup > .btn > .caret {
    border-top-color: #000 !important;
  }
  .label {
    border: 1px solid #000;
  }
  .table {
    border-collapse: collapse !important;
  }
  .table td,
  .table th {
    background-color: #fff !important;
  }
  .table-bordered th,
  .table-bordered td {
    border: 1px solid #ddd !important;
  }
}
@font-face {
  font-family: 'Glyphicons Halflings';
  src: url('../components/bootstrap/fonts/glyphicons-halflings-regular.eot');
  src: url('../components/bootstrap/fonts/glyphicons-halflings-regular.eot?#iefix') format('embedded-opentype'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.woff2') format('woff2'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.woff') format('woff'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.ttf') format('truetype'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.svg#glyphicons_halflingsregular') format('svg');
}
.glyphicon {
  position: relative;
  top: 1px;
  display: inline-block;
  font-family: 'Glyphicons Halflings';
  font-style: normal;
  font-weight: normal;
  line-height: 1;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
.glyphicon-asterisk:before {
  content: "\002a";
}
.glyphicon-plus:before {
  content: "\002b";
}
.glyphicon-euro:before,
.glyphicon-eur:before {
  content: "\20ac";
}
.glyphicon-minus:before {
  content: "\2212";
}
.glyphicon-cloud:before {
  content: "\2601";
}
.glyphicon-envelope:before {
  content: "\2709";
}
.glyphicon-pencil:before {
  content: "\270f";
}
.glyphicon-glass:before {
  content: "\e001";
}
.glyphicon-music:before {
  content: "\e002";
}
.glyphicon-search:before {
  content: "\e003";
}
.glyphicon-heart:before {
  content: "\e005";
}
.glyphicon-star:before {
  content: "\e006";
}
.glyphicon-star-empty:before {
  content: "\e007";
}
.glyphicon-user:before {
  content: "\e008";
}
.glyphicon-film:before {
  content: "\e009";
}
.glyphicon-th-large:before {
  content: "\e010";
}
.glyphicon-th:before {
  content: "\e011";
}
.glyphicon-th-list:before {
  content: "\e012";
}
.glyphicon-ok:before {
  content: "\e013";
}
.glyphicon-remove:before {
  content: "\e014";
}
.glyphicon-zoom-in:before {
  content: "\e015";
}
.glyphicon-zoom-out:before {
  content: "\e016";
}
.glyphicon-off:before {
  content: "\e017";
}
.glyphicon-signal:before {
  content: "\e018";
}
.glyphicon-cog:before {
  content: "\e019";
}
.glyphicon-trash:before {
  content: "\e020";
}
.glyphicon-home:before {
  content: "\e021";
}
.glyphicon-file:before {
  content: "\e022";
}
.glyphicon-time:before {
  content: "\e023";
}
.glyphicon-road:before {
  content: "\e024";
}
.glyphicon-download-alt:before {
  content: "\e025";
}
.glyphicon-download:before {
  content: "\e026";
}
.glyphicon-upload:before {
  content: "\e027";
}
.glyphicon-inbox:before {
  content: "\e028";
}
.glyphicon-play-circle:before {
  content: "\e029";
}
.glyphicon-repeat:before {
  content: "\e030";
}
.glyphicon-refresh:before {
  content: "\e031";
}
.glyphicon-list-alt:before {
  content: "\e032";
}
.glyphicon-lock:before {
  content: "\e033";
}
.glyphicon-flag:before {
  content: "\e034";
}
.glyphicon-headphones:before {
  content: "\e035";
}
.glyphicon-volume-off:before {
  content: "\e036";
}
.glyphicon-volume-down:before {
  content: "\e037";
}
.glyphicon-volume-up:before {
  content: "\e038";
}
.glyphicon-qrcode:before {
  content: "\e039";
}
.glyphicon-barcode:before {
  content: "\e040";
}
.glyphicon-tag:before {
  content: "\e041";
}
.glyphicon-tags:before {
  content: "\e042";
}
.glyphicon-book:before {
  content: "\e043";
}
.glyphicon-bookmark:before {
  content: "\e044";
}
.glyphicon-print:before {
  content: "\e045";
}
.glyphicon-camera:before {
  content: "\e046";
}
.glyphicon-font:before {
  content: "\e047";
}
.glyphicon-bold:before {
  content: "\e048";
}
.glyphicon-italic:before {
  content: "\e049";
}
.glyphicon-text-height:before {
  content: "\e050";
}
.glyphicon-text-width:before {
  content: "\e051";
}
.glyphicon-align-left:before {
  content: "\e052";
}
.glyphicon-align-center:before {
  content: "\e053";
}
.glyphicon-align-right:before {
  content: "\e054";
}
.glyphicon-align-justify:before {
  content: "\e055";
}
.glyphicon-list:before {
  content: "\e056";
}
.glyphicon-indent-left:before {
  content: "\e057";
}
.glyphicon-indent-right:before {
  content: "\e058";
}
.glyphicon-facetime-video:before {
  content: "\e059";
}
.glyphicon-picture:before {
  content: "\e060";
}
.glyphicon-map-marker:before {
  content: "\e062";
}
.glyphicon-adjust:before {
  content: "\e063";
}
.glyphicon-tint:before {
  content: "\e064";
}
.glyphicon-edit:before {
  content: "\e065";
}
.glyphicon-share:before {
  content: "\e066";
}
.glyphicon-check:before {
  content: "\e067";
}
.glyphicon-move:before {
  content: "\e068";
}
.glyphicon-step-backward:before {
  content: "\e069";
}
.glyphicon-fast-backward:before {
  content: "\e070";
}
.glyphicon-backward:before {
  content: "\e071";
}
.glyphicon-play:before {
  content: "\e072";
}
.glyphicon-pause:before {
  content: "\e073";
}
.glyphicon-stop:before {
  content: "\e074";
}
.glyphicon-forward:before {
  content: "\e075";
}
.glyphicon-fast-forward:before {
  content: "\e076";
}
.glyphicon-step-forward:before {
  content: "\e077";
}
.glyphicon-eject:before {
  content: "\e078";
}
.glyphicon-chevron-left:before {
  content: "\e079";
}
.glyphicon-chevron-right:before {
  content: "\e080";
}
.glyphicon-plus-sign:before {
  content: "\e081";
}
.glyphicon-minus-sign:before {
  content: "\e082";
}
.glyphicon-remove-sign:before {
  content: "\e083";
}
.glyphicon-ok-sign:before {
  content: "\e084";
}
.glyphicon-question-sign:before {
  content: "\e085";
}
.glyphicon-info-sign:before {
  content: "\e086";
}
.glyphicon-screenshot:before {
  content: "\e087";
}
.glyphicon-remove-circle:before {
  content: "\e088";
}
.glyphicon-ok-circle:before {
  content: "\e089";
}
.glyphicon-ban-circle:before {
  content: "\e090";
}
.glyphicon-arrow-left:before {
  content: "\e091";
}
.glyphicon-arrow-right:before {
  content: "\e092";
}
.glyphicon-arrow-up:before {
  content: "\e093";
}
.glyphicon-arrow-down:before {
  content: "\e094";
}
.glyphicon-share-alt:before {
  content: "\e095";
}
.glyphicon-resize-full:before {
  content: "\e096";
}
.glyphicon-resize-small:before {
  content: "\e097";
}
.glyphicon-exclamation-sign:before {
  content: "\e101";
}
.glyphicon-gift:before {
  content: "\e102";
}
.glyphicon-leaf:before {
  content: "\e103";
}
.glyphicon-fire:before {
  content: "\e104";
}
.glyphicon-eye-open:before {
  content: "\e105";
}
.glyphicon-eye-close:before {
  content: "\e106";
}
.glyphicon-warning-sign:before {
  content: "\e107";
}
.glyphicon-plane:before {
  content: "\e108";
}
.glyphicon-calendar:before {
  content: "\e109";
}
.glyphicon-random:before {
  content: "\e110";
}
.glyphicon-comment:before {
  content: "\e111";
}
.glyphicon-magnet:before {
  content: "\e112";
}
.glyphicon-chevron-up:before {
  content: "\e113";
}
.glyphicon-chevron-down:before {
  content: "\e114";
}
.glyphicon-retweet:before {
  content: "\e115";
}
.glyphicon-shopping-cart:before {
  content: "\e116";
}
.glyphicon-folder-close:before {
  content: "\e117";
}
.glyphicon-folder-open:before {
  content: "\e118";
}
.glyphicon-resize-vertical:before {
  content: "\e119";
}
.glyphicon-resize-horizontal:before {
  content: "\e120";
}
.glyphicon-hdd:before {
  content: "\e121";
}
.glyphicon-bullhorn:before {
  content: "\e122";
}
.glyphicon-bell:before {
  content: "\e123";
}
.glyphicon-certificate:before {
  content: "\e124";
}
.glyphicon-thumbs-up:before {
  content: "\e125";
}
.glyphicon-thumbs-down:before {
  content: "\e126";
}
.glyphicon-hand-right:before {
  content: "\e127";
}
.glyphicon-hand-left:before {
  content: "\e128";
}
.glyphicon-hand-up:before {
  content: "\e129";
}
.glyphicon-hand-down:before {
  content: "\e130";
}
.glyphicon-circle-arrow-right:before {
  content: "\e131";
}
.glyphicon-circle-arrow-left:before {
  content: "\e132";
}
.glyphicon-circle-arrow-up:before {
  content: "\e133";
}
.glyphicon-circle-arrow-down:before {
  content: "\e134";
}
.glyphicon-globe:before {
  content: "\e135";
}
.glyphicon-wrench:before {
  content: "\e136";
}
.glyphicon-tasks:before {
  content: "\e137";
}
.glyphicon-filter:before {
  content: "\e138";
}
.glyphicon-briefcase:before {
  content: "\e139";
}
.glyphicon-fullscreen:before {
  content: "\e140";
}
.glyphicon-dashboard:before {
  content: "\e141";
}
.glyphicon-paperclip:before {
  content: "\e142";
}
.glyphicon-heart-empty:before {
  content: "\e143";
}
.glyphicon-link:before {
  content: "\e144";
}
.glyphicon-phone:before {
  content: "\e145";
}
.glyphicon-pushpin:before {
  content: "\e146";
}
.glyphicon-usd:before {
  content: "\e148";
}
.glyphicon-gbp:before {
  content: "\e149";
}
.glyphicon-sort:before {
  content: "\e150";
}
.glyphicon-sort-by-alphabet:before {
  content: "\e151";
}
.glyphicon-sort-by-alphabet-alt:before {
  content: "\e152";
}
.glyphicon-sort-by-order:before {
  content: "\e153";
}
.glyphicon-sort-by-order-alt:before {
  content: "\e154";
}
.glyphicon-sort-by-attributes:before {
  content: "\e155";
}
.glyphicon-sort-by-attributes-alt:before {
  content: "\e156";
}
.glyphicon-unchecked:before {
  content: "\e157";
}
.glyphicon-expand:before {
  content: "\e158";
}
.glyphicon-collapse-down:before {
  content: "\e159";
}
.glyphicon-collapse-up:before {
  content: "\e160";
}
.glyphicon-log-in:before {
  content: "\e161";
}
.glyphicon-flash:before {
  content: "\e162";
}
.glyphicon-log-out:before {
  content: "\e163";
}
.glyphicon-new-window:before {
  content: "\e164";
}
.glyphicon-record:before {
  content: "\e165";
}
.glyphicon-save:before {
  content: "\e166";
}
.glyphicon-open:before {
  content: "\e167";
}
.glyphicon-saved:before {
  content: "\e168";
}
.glyphicon-import:before {
  content: "\e169";
}
.glyphicon-export:before {
  content: "\e170";
}
.glyphicon-send:before {
  content: "\e171";
}
.glyphicon-floppy-disk:before {
  content: "\e172";
}
.glyphicon-floppy-saved:before {
  content: "\e173";
}
.glyphicon-floppy-remove:before {
  content: "\e174";
}
.glyphicon-floppy-save:before {
  content: "\e175";
}
.glyphicon-floppy-open:before {
  content: "\e176";
}
.glyphicon-credit-card:before {
  content: "\e177";
}
.glyphicon-transfer:before {
  content: "\e178";
}
.glyphicon-cutlery:before {
  content: "\e179";
}
.glyphicon-header:before {
  content: "\e180";
}
.glyphicon-compressed:before {
  content: "\e181";
}
.glyphicon-earphone:before {
  content: "\e182";
}
.glyphicon-phone-alt:before {
  content: "\e183";
}
.glyphicon-tower:before {
  content: "\e184";
}
.glyphicon-stats:before {
  content: "\e185";
}
.glyphicon-sd-video:before {
  content: "\e186";
}
.glyphicon-hd-video:before {
  content: "\e187";
}
.glyphicon-subtitles:before {
  content: "\e188";
}
.glyphicon-sound-stereo:before {
  content: "\e189";
}
.glyphicon-sound-dolby:before {
  content: "\e190";
}
.glyphicon-sound-5-1:before {
  content: "\e191";
}
.glyphicon-sound-6-1:before {
  content: "\e192";
}
.glyphicon-sound-7-1:before {
  content: "\e193";
}
.glyphicon-copyright-mark:before {
  content: "\e194";
}
.glyphicon-registration-mark:before {
  content: "\e195";
}
.glyphicon-cloud-download:before {
  content: "\e197";
}
.glyphicon-cloud-upload:before {
  content: "\e198";
}
.glyphicon-tree-conifer:before {
  content: "\e199";
}
.glyphicon-tree-deciduous:before {
  content: "\e200";
}
.glyphicon-cd:before {
  content: "\e201";
}
.glyphicon-save-file:before {
  content: "\e202";
}
.glyphicon-open-file:before {
  content: "\e203";
}
.glyphicon-level-up:before {
  content: "\e204";
}
.glyphicon-copy:before {
  content: "\e205";
}
.glyphicon-paste:before {
  content: "\e206";
}
.glyphicon-alert:before {
  content: "\e209";
}
.glyphicon-equalizer:before {
  content: "\e210";
}
.glyphicon-king:before {
  content: "\e211";
}
.glyphicon-queen:before {
  content: "\e212";
}
.glyphicon-pawn:before {
  content: "\e213";
}
.glyphicon-bishop:before {
  content: "\e214";
}
.glyphicon-knight:before {
  content: "\e215";
}
.glyphicon-baby-formula:before {
  content: "\e216";
}
.glyphicon-tent:before {
  content: "\26fa";
}
.glyphicon-blackboard:before {
  content: "\e218";
}
.glyphicon-bed:before {
  content: "\e219";
}
.glyphicon-apple:before {
  content: "\f8ff";
}
.glyphicon-erase:before {
  content: "\e221";
}
.glyphicon-hourglass:before {
  content: "\231b";
}
.glyphicon-lamp:before {
  content: "\e223";
}
.glyphicon-duplicate:before {
  content: "\e224";
}
.glyphicon-piggy-bank:before {
  content: "\e225";
}
.glyphicon-scissors:before {
  content: "\e226";
}
.glyphicon-bitcoin:before {
  content: "\e227";
}
.glyphicon-btc:before {
  content: "\e227";
}
.glyphicon-xbt:before {
  content: "\e227";
}
.glyphicon-yen:before {
  content: "\00a5";
}
.glyphicon-jpy:before {
  content: "\00a5";
}
.glyphicon-ruble:before {
  content: "\20bd";
}
.glyphicon-rub:before {
  content: "\20bd";
}
.glyphicon-scale:before {
  content: "\e230";
}
.glyphicon-ice-lolly:before {
  content: "\e231";
}
.glyphicon-ice-lolly-tasted:before {
  content: "\e232";
}
.glyphicon-education:before {
  content: "\e233";
}
.glyphicon-option-horizontal:before {
  content: "\e234";
}
.glyphicon-option-vertical:before {
  content: "\e235";
}
.glyphicon-menu-hamburger:before {
  content: "\e236";
}
.glyphicon-modal-window:before {
  content: "\e237";
}
.glyphicon-oil:before {
  content: "\e238";
}
.glyphicon-grain:before {
  content: "\e239";
}
.glyphicon-sunglasses:before {
  content: "\e240";
}
.glyphicon-text-size:before {
  content: "\e241";
}
.glyphicon-text-color:before {
  content: "\e242";
}
.glyphicon-text-background:before {
  content: "\e243";
}
.glyphicon-object-align-top:before {
  content: "\e244";
}
.glyphicon-object-align-bottom:before {
  content: "\e245";
}
.glyphicon-object-align-horizontal:before {
  content: "\e246";
}
.glyphicon-object-align-left:before {
  content: "\e247";
}
.glyphicon-object-align-vertical:before {
  content: "\e248";
}
.glyphicon-object-align-right:before {
  content: "\e249";
}
.glyphicon-triangle-right:before {
  content: "\e250";
}
.glyphicon-triangle-left:before {
  content: "\e251";
}
.glyphicon-triangle-bottom:before {
  content: "\e252";
}
.glyphicon-triangle-top:before {
  content: "\e253";
}
.glyphicon-console:before {
  content: "\e254";
}
.glyphicon-superscript:before {
  content: "\e255";
}
.glyphicon-subscript:before {
  content: "\e256";
}
.glyphicon-menu-left:before {
  content: "\e257";
}
.glyphicon-menu-right:before {
  content: "\e258";
}
.glyphicon-menu-down:before {
  content: "\e259";
}
.glyphicon-menu-up:before {
  content: "\e260";
}
* {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
*:before,
*:after {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
html {
  font-size: 10px;
  -webkit-tap-highlight-color: rgba(0, 0, 0, 0);
}
body {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 13px;
  line-height: 1.42857143;
  color: #000;
  background-color: #fff;
}
input,
button,
select,
textarea {
  font-family: inherit;
  font-size: inherit;
  line-height: inherit;
}
a {
  color: #337ab7;
  text-decoration: none;
}
a:hover,
a:focus {
  color: #23527c;
  text-decoration: underline;
}
a:focus {
  outline: thin dotted;
  outline: 5px auto -webkit-focus-ring-color;
  outline-offset: -2px;
}
figure {
  margin: 0;
}
img {
  vertical-align: middle;
}
.img-responsive,
.thumbnail > img,
.thumbnail a > img,
.carousel-inner > .item > img,
.carousel-inner > .item > a > img {
  display: block;
  max-width: 100%;
  height: auto;
}
.img-rounded {
  border-radius: 3px;
}
.img-thumbnail {
  padding: 4px;
  line-height: 1.42857143;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 2px;
  -webkit-transition: all 0.2s ease-in-out;
  -o-transition: all 0.2s ease-in-out;
  transition: all 0.2s ease-in-out;
  display: inline-block;
  max-width: 100%;
  height: auto;
}
.img-circle {
  border-radius: 50%;
}
hr {
  margin-top: 18px;
  margin-bottom: 18px;
  border: 0;
  border-top: 1px solid #eeeeee;
}
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  margin: -1px;
  padding: 0;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}
.sr-only-focusable:active,
.sr-only-focusable:focus {
  position: static;
  width: auto;
  height: auto;
  margin: 0;
  overflow: visible;
  clip: auto;
}
[role="button"] {
  cursor: pointer;
}
h1,
h2,
h3,
h4,
h5,
h6,
.h1,
.h2,
.h3,
.h4,
.h5,
.h6 {
  font-family: inherit;
  font-weight: 500;
  line-height: 1.1;
  color: inherit;
}
h1 small,
h2 small,
h3 small,
h4 small,
h5 small,
h6 small,
.h1 small,
.h2 small,
.h3 small,
.h4 small,
.h5 small,
.h6 small,
h1 .small,
h2 .small,
h3 .small,
h4 .small,
h5 .small,
h6 .small,
.h1 .small,
.h2 .small,
.h3 .small,
.h4 .small,
.h5 .small,
.h6 .small {
  font-weight: normal;
  line-height: 1;
  color: #777777;
}
h1,
.h1,
h2,
.h2,
h3,
.h3 {
  margin-top: 18px;
  margin-bottom: 9px;
}
h1 small,
.h1 small,
h2 small,
.h2 small,
h3 small,
.h3 small,
h1 .small,
.h1 .small,
h2 .small,
.h2 .small,
h3 .small,
.h3 .small {
  font-size: 65%;
}
h4,
.h4,
h5,
.h5,
h6,
.h6 {
  margin-top: 9px;
  margin-bottom: 9px;
}
h4 small,
.h4 small,
h5 small,
.h5 small,
h6 small,
.h6 small,
h4 .small,
.h4 .small,
h5 .small,
.h5 .small,
h6 .small,
.h6 .small {
  font-size: 75%;
}
h1,
.h1 {
  font-size: 33px;
}
h2,
.h2 {
  font-size: 27px;
}
h3,
.h3 {
  font-size: 23px;
}
h4,
.h4 {
  font-size: 17px;
}
h5,
.h5 {
  font-size: 13px;
}
h6,
.h6 {
  font-size: 12px;
}
p {
  margin: 0 0 9px;
}
.lead {
  margin-bottom: 18px;
  font-size: 14px;
  font-weight: 300;
  line-height: 1.4;
}
@media (min-width: 768px) {
  .lead {
    font-size: 19.5px;
  }
}
small,
.small {
  font-size: 92%;
}
mark,
.mark {
  background-color: #fcf8e3;
  padding: .2em;
}
.text-left {
  text-align: left;
}
.text-right {
  text-align: right;
}
.text-center {
  text-align: center;
}
.text-justify {
  text-align: justify;
}
.text-nowrap {
  white-space: nowrap;
}
.text-lowercase {
  text-transform: lowercase;
}
.text-uppercase {
  text-transform: uppercase;
}
.text-capitalize {
  text-transform: capitalize;
}
.text-muted {
  color: #777777;
}
.text-primary {
  color: #337ab7;
}
a.text-primary:hover,
a.text-primary:focus {
  color: #286090;
}
.text-success {
  color: #3c763d;
}
a.text-success:hover,
a.text-success:focus {
  color: #2b542c;
}
.text-info {
  color: #31708f;
}
a.text-info:hover,
a.text-info:focus {
  color: #245269;
}
.text-warning {
  color: #8a6d3b;
}
a.text-warning:hover,
a.text-warning:focus {
  color: #66512c;
}
.text-danger {
  color: #a94442;
}
a.text-danger:hover,
a.text-danger:focus {
  color: #843534;
}
.bg-primary {
  color: #fff;
  background-color: #337ab7;
}
a.bg-primary:hover,
a.bg-primary:focus {
  background-color: #286090;
}
.bg-success {
  background-color: #dff0d8;
}
a.bg-success:hover,
a.bg-success:focus {
  background-color: #c1e2b3;
}
.bg-info {
  background-color: #d9edf7;
}
a.bg-info:hover,
a.bg-info:focus {
  background-color: #afd9ee;
}
.bg-warning {
  background-color: #fcf8e3;
}
a.bg-warning:hover,
a.bg-warning:focus {
  background-color: #f7ecb5;
}
.bg-danger {
  background-color: #f2dede;
}
a.bg-danger:hover,
a.bg-danger:focus {
  background-color: #e4b9b9;
}
.page-header {
  padding-bottom: 8px;
  margin: 36px 0 18px;
  border-bottom: 1px solid #eeeeee;
}
ul,
ol {
  margin-top: 0;
  margin-bottom: 9px;
}
ul ul,
ol ul,
ul ol,
ol ol {
  margin-bottom: 0;
}
.list-unstyled {
  padding-left: 0;
  list-style: none;
}
.list-inline {
  padding-left: 0;
  list-style: none;
  margin-left: -5px;
}
.list-inline > li {
  display: inline-block;
  padding-left: 5px;
  padding-right: 5px;
}
dl {
  margin-top: 0;
  margin-bottom: 18px;
}
dt,
dd {
  line-height: 1.42857143;
}
dt {
  font-weight: bold;
}
dd {
  margin-left: 0;
}
@media (min-width: 541px) {
  .dl-horizontal dt {
    float: left;
    width: 160px;
    clear: left;
    text-align: right;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .dl-horizontal dd {
    margin-left: 180px;
  }
}
abbr[title],
abbr[data-original-title] {
  cursor: help;
  border-bottom: 1px dotted #777777;
}
.initialism {
  font-size: 90%;
  text-transform: uppercase;
}
blockquote {
  padding: 9px 18px;
  margin: 0 0 18px;
  font-size: inherit;
  border-left: 5px solid #eeeeee;
}
blockquote p:last-child,
blockquote ul:last-child,
blockquote ol:last-child {
  margin-bottom: 0;
}
blockquote footer,
blockquote small,
blockquote .small {
  display: block;
  font-size: 80%;
  line-height: 1.42857143;
  color: #777777;
}
blockquote footer:before,
blockquote small:before,
blockquote .small:before {
  content: '\2014 \00A0';
}
.blockquote-reverse,
blockquote.pull-right {
  padding-right: 15px;
  padding-left: 0;
  border-right: 5px solid #eeeeee;
  border-left: 0;
  text-align: right;
}
.blockquote-reverse footer:before,
blockquote.pull-right footer:before,
.blockquote-reverse small:before,
blockquote.pull-right small:before,
.blockquote-reverse .small:before,
blockquote.pull-right .small:before {
  content: '';
}
.blockquote-reverse footer:after,
blockquote.pull-right footer:after,
.blockquote-reverse small:after,
blockquote.pull-right small:after,
.blockquote-reverse .small:after,
blockquote.pull-right .small:after {
  content: '\00A0 \2014';
}
address {
  margin-bottom: 18px;
  font-style: normal;
  line-height: 1.42857143;
}
code,
kbd,
pre,
samp {
  font-family: monospace;
}
code {
  padding: 2px 4px;
  font-size: 90%;
  color: #c7254e;
  background-color: #f9f2f4;
  border-radius: 2px;
}
kbd {
  padding: 2px 4px;
  font-size: 90%;
  color: #888;
  background-color: transparent;
  border-radius: 1px;
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.25);
}
kbd kbd {
  padding: 0;
  font-size: 100%;
  font-weight: bold;
  box-shadow: none;
}
pre {
  display: block;
  padding: 8.5px;
  margin: 0 0 9px;
  font-size: 12px;
  line-height: 1.42857143;
  word-break: break-all;
  word-wrap: break-word;
  color: #333333;
  background-color: #f5f5f5;
  border: 1px solid #ccc;
  border-radius: 2px;
}
pre code {
  padding: 0;
  font-size: inherit;
  color: inherit;
  white-space: pre-wrap;
  background-color: transparent;
  border-radius: 0;
}
.pre-scrollable {
  max-height: 340px;
  overflow-y: scroll;
}
.container {
  margin-right: auto;
  margin-left: auto;
  padding-left: 0px;
  padding-right: 0px;
}
@media (min-width: 768px) {
  .container {
    width: 768px;
  }
}
@media (min-width: 992px) {
  .container {
    width: 940px;
  }
}
@media (min-width: 1200px) {
  .container {
    width: 1140px;
  }
}
.container-fluid {
  margin-right: auto;
  margin-left: auto;
  padding-left: 0px;
  padding-right: 0px;
}
.row {
  margin-left: 0px;
  margin-right: 0px;
}
.col-xs-1, .col-sm-1, .col-md-1, .col-lg-1, .col-xs-2, .col-sm-2, .col-md-2, .col-lg-2, .col-xs-3, .col-sm-3, .col-md-3, .col-lg-3, .col-xs-4, .col-sm-4, .col-md-4, .col-lg-4, .col-xs-5, .col-sm-5, .col-md-5, .col-lg-5, .col-xs-6, .col-sm-6, .col-md-6, .col-lg-6, .col-xs-7, .col-sm-7, .col-md-7, .col-lg-7, .col-xs-8, .col-sm-8, .col-md-8, .col-lg-8, .col-xs-9, .col-sm-9, .col-md-9, .col-lg-9, .col-xs-10, .col-sm-10, .col-md-10, .col-lg-10, .col-xs-11, .col-sm-11, .col-md-11, .col-lg-11, .col-xs-12, .col-sm-12, .col-md-12, .col-lg-12 {
  position: relative;
  min-height: 1px;
  padding-left: 0px;
  padding-right: 0px;
}
.col-xs-1, .col-xs-2, .col-xs-3, .col-xs-4, .col-xs-5, .col-xs-6, .col-xs-7, .col-xs-8, .col-xs-9, .col-xs-10, .col-xs-11, .col-xs-12 {
  float: left;
}
.col-xs-12 {
  width: 100%;
}
.col-xs-11 {
  width: 91.66666667%;
}
.col-xs-10 {
  width: 83.33333333%;
}
.col-xs-9 {
  width: 75%;
}
.col-xs-8 {
  width: 66.66666667%;
}
.col-xs-7 {
  width: 58.33333333%;
}
.col-xs-6 {
  width: 50%;
}
.col-xs-5 {
  width: 41.66666667%;
}
.col-xs-4 {
  width: 33.33333333%;
}
.col-xs-3 {
  width: 25%;
}
.col-xs-2 {
  width: 16.66666667%;
}
.col-xs-1 {
  width: 8.33333333%;
}
.col-xs-pull-12 {
  right: 100%;
}
.col-xs-pull-11 {
  right: 91.66666667%;
}
.col-xs-pull-10 {
  right: 83.33333333%;
}
.col-xs-pull-9 {
  right: 75%;
}
.col-xs-pull-8 {
  right: 66.66666667%;
}
.col-xs-pull-7 {
  right: 58.33333333%;
}
.col-xs-pull-6 {
  right: 50%;
}
.col-xs-pull-5 {
  right: 41.66666667%;
}
.col-xs-pull-4 {
  right: 33.33333333%;
}
.col-xs-pull-3 {
  right: 25%;
}
.col-xs-pull-2 {
  right: 16.66666667%;
}
.col-xs-pull-1 {
  right: 8.33333333%;
}
.col-xs-pull-0 {
  right: auto;
}
.col-xs-push-12 {
  left: 100%;
}
.col-xs-push-11 {
  left: 91.66666667%;
}
.col-xs-push-10 {
  left: 83.33333333%;
}
.col-xs-push-9 {
  left: 75%;
}
.col-xs-push-8 {
  left: 66.66666667%;
}
.col-xs-push-7 {
  left: 58.33333333%;
}
.col-xs-push-6 {
  left: 50%;
}
.col-xs-push-5 {
  left: 41.66666667%;
}
.col-xs-push-4 {
  left: 33.33333333%;
}
.col-xs-push-3 {
  left: 25%;
}
.col-xs-push-2 {
  left: 16.66666667%;
}
.col-xs-push-1 {
  left: 8.33333333%;
}
.col-xs-push-0 {
  left: auto;
}
.col-xs-offset-12 {
  margin-left: 100%;
}
.col-xs-offset-11 {
  margin-left: 91.66666667%;
}
.col-xs-offset-10 {
  margin-left: 83.33333333%;
}
.col-xs-offset-9 {
  margin-left: 75%;
}
.col-xs-offset-8 {
  margin-left: 66.66666667%;
}
.col-xs-offset-7 {
  margin-left: 58.33333333%;
}
.col-xs-offset-6 {
  margin-left: 50%;
}
.col-xs-offset-5 {
  margin-left: 41.66666667%;
}
.col-xs-offset-4 {
  margin-left: 33.33333333%;
}
.col-xs-offset-3 {
  margin-left: 25%;
}
.col-xs-offset-2 {
  margin-left: 16.66666667%;
}
.col-xs-offset-1 {
  margin-left: 8.33333333%;
}
.col-xs-offset-0 {
  margin-left: 0%;
}
@media (min-width: 768px) {
  .col-sm-1, .col-sm-2, .col-sm-3, .col-sm-4, .col-sm-5, .col-sm-6, .col-sm-7, .col-sm-8, .col-sm-9, .col-sm-10, .col-sm-11, .col-sm-12 {
    float: left;
  }
  .col-sm-12 {
    width: 100%;
  }
  .col-sm-11 {
    width: 91.66666667%;
  }
  .col-sm-10 {
    width: 83.33333333%;
  }
  .col-sm-9 {
    width: 75%;
  }
  .col-sm-8 {
    width: 66.66666667%;
  }
  .col-sm-7 {
    width: 58.33333333%;
  }
  .col-sm-6 {
    width: 50%;
  }
  .col-sm-5 {
    width: 41.66666667%;
  }
  .col-sm-4 {
    width: 33.33333333%;
  }
  .col-sm-3 {
    width: 25%;
  }
  .col-sm-2 {
    width: 16.66666667%;
  }
  .col-sm-1 {
    width: 8.33333333%;
  }
  .col-sm-pull-12 {
    right: 100%;
  }
  .col-sm-pull-11 {
    right: 91.66666667%;
  }
  .col-sm-pull-10 {
    right: 83.33333333%;
  }
  .col-sm-pull-9 {
    right: 75%;
  }
  .col-sm-pull-8 {
    right: 66.66666667%;
  }
  .col-sm-pull-7 {
    right: 58.33333333%;
  }
  .col-sm-pull-6 {
    right: 50%;
  }
  .col-sm-pull-5 {
    right: 41.66666667%;
  }
  .col-sm-pull-4 {
    right: 33.33333333%;
  }
  .col-sm-pull-3 {
    right: 25%;
  }
  .col-sm-pull-2 {
    right: 16.66666667%;
  }
  .col-sm-pull-1 {
    right: 8.33333333%;
  }
  .col-sm-pull-0 {
    right: auto;
  }
  .col-sm-push-12 {
    left: 100%;
  }
  .col-sm-push-11 {
    left: 91.66666667%;
  }
  .col-sm-push-10 {
    left: 83.33333333%;
  }
  .col-sm-push-9 {
    left: 75%;
  }
  .col-sm-push-8 {
    left: 66.66666667%;
  }
  .col-sm-push-7 {
    left: 58.33333333%;
  }
  .col-sm-push-6 {
    left: 50%;
  }
  .col-sm-push-5 {
    left: 41.66666667%;
  }
  .col-sm-push-4 {
    left: 33.33333333%;
  }
  .col-sm-push-3 {
    left: 25%;
  }
  .col-sm-push-2 {
    left: 16.66666667%;
  }
  .col-sm-push-1 {
    left: 8.33333333%;
  }
  .col-sm-push-0 {
    left: auto;
  }
  .col-sm-offset-12 {
    margin-left: 100%;
  }
  .col-sm-offset-11 {
    margin-left: 91.66666667%;
  }
  .col-sm-offset-10 {
    margin-left: 83.33333333%;
  }
  .col-sm-offset-9 {
    margin-left: 75%;
  }
  .col-sm-offset-8 {
    margin-left: 66.66666667%;
  }
  .col-sm-offset-7 {
    margin-left: 58.33333333%;
  }
  .col-sm-offset-6 {
    margin-left: 50%;
  }
  .col-sm-offset-5 {
    margin-left: 41.66666667%;
  }
  .col-sm-offset-4 {
    margin-left: 33.33333333%;
  }
  .col-sm-offset-3 {
    margin-left: 25%;
  }
  .col-sm-offset-2 {
    margin-left: 16.66666667%;
  }
  .col-sm-offset-1 {
    margin-left: 8.33333333%;
  }
  .col-sm-offset-0 {
    margin-left: 0%;
  }
}
@media (min-width: 992px) {
  .col-md-1, .col-md-2, .col-md-3, .col-md-4, .col-md-5, .col-md-6, .col-md-7, .col-md-8, .col-md-9, .col-md-10, .col-md-11, .col-md-12 {
    float: left;
  }
  .col-md-12 {
    width: 100%;
  }
  .col-md-11 {
    width: 91.66666667%;
  }
  .col-md-10 {
    width: 83.33333333%;
  }
  .col-md-9 {
    width: 75%;
  }
  .col-md-8 {
    width: 66.66666667%;
  }
  .col-md-7 {
    width: 58.33333333%;
  }
  .col-md-6 {
    width: 50%;
  }
  .col-md-5 {
    width: 41.66666667%;
  }
  .col-md-4 {
    width: 33.33333333%;
  }
  .col-md-3 {
    width: 25%;
  }
  .col-md-2 {
    width: 16.66666667%;
  }
  .col-md-1 {
    width: 8.33333333%;
  }
  .col-md-pull-12 {
    right: 100%;
  }
  .col-md-pull-11 {
    right: 91.66666667%;
  }
  .col-md-pull-10 {
    right: 83.33333333%;
  }
  .col-md-pull-9 {
    right: 75%;
  }
  .col-md-pull-8 {
    right: 66.66666667%;
  }
  .col-md-pull-7 {
    right: 58.33333333%;
  }
  .col-md-pull-6 {
    right: 50%;
  }
  .col-md-pull-5 {
    right: 41.66666667%;
  }
  .col-md-pull-4 {
    right: 33.33333333%;
  }
  .col-md-pull-3 {
    right: 25%;
  }
  .col-md-pull-2 {
    right: 16.66666667%;
  }
  .col-md-pull-1 {
    right: 8.33333333%;
  }
  .col-md-pull-0 {
    right: auto;
  }
  .col-md-push-12 {
    left: 100%;
  }
  .col-md-push-11 {
    left: 91.66666667%;
  }
  .col-md-push-10 {
    left: 83.33333333%;
  }
  .col-md-push-9 {
    left: 75%;
  }
  .col-md-push-8 {
    left: 66.66666667%;
  }
  .col-md-push-7 {
    left: 58.33333333%;
  }
  .col-md-push-6 {
    left: 50%;
  }
  .col-md-push-5 {
    left: 41.66666667%;
  }
  .col-md-push-4 {
    left: 33.33333333%;
  }
  .col-md-push-3 {
    left: 25%;
  }
  .col-md-push-2 {
    left: 16.66666667%;
  }
  .col-md-push-1 {
    left: 8.33333333%;
  }
  .col-md-push-0 {
    left: auto;
  }
  .col-md-offset-12 {
    margin-left: 100%;
  }
  .col-md-offset-11 {
    margin-left: 91.66666667%;
  }
  .col-md-offset-10 {
    margin-left: 83.33333333%;
  }
  .col-md-offset-9 {
    margin-left: 75%;
  }
  .col-md-offset-8 {
    margin-left: 66.66666667%;
  }
  .col-md-offset-7 {
    margin-left: 58.33333333%;
  }
  .col-md-offset-6 {
    margin-left: 50%;
  }
  .col-md-offset-5 {
    margin-left: 41.66666667%;
  }
  .col-md-offset-4 {
    margin-left: 33.33333333%;
  }
  .col-md-offset-3 {
    margin-left: 25%;
  }
  .col-md-offset-2 {
    margin-left: 16.66666667%;
  }
  .col-md-offset-1 {
    margin-left: 8.33333333%;
  }
  .col-md-offset-0 {
    margin-left: 0%;
  }
}
@media (min-width: 1200px) {
  .col-lg-1, .col-lg-2, .col-lg-3, .col-lg-4, .col-lg-5, .col-lg-6, .col-lg-7, .col-lg-8, .col-lg-9, .col-lg-10, .col-lg-11, .col-lg-12 {
    float: left;
  }
  .col-lg-12 {
    width: 100%;
  }
  .col-lg-11 {
    width: 91.66666667%;
  }
  .col-lg-10 {
    width: 83.33333333%;
  }
  .col-lg-9 {
    width: 75%;
  }
  .col-lg-8 {
    width: 66.66666667%;
  }
  .col-lg-7 {
    width: 58.33333333%;
  }
  .col-lg-6 {
    width: 50%;
  }
  .col-lg-5 {
    width: 41.66666667%;
  }
  .col-lg-4 {
    width: 33.33333333%;
  }
  .col-lg-3 {
    width: 25%;
  }
  .col-lg-2 {
    width: 16.66666667%;
  }
  .col-lg-1 {
    width: 8.33333333%;
  }
  .col-lg-pull-12 {
    right: 100%;
  }
  .col-lg-pull-11 {
    right: 91.66666667%;
  }
  .col-lg-pull-10 {
    right: 83.33333333%;
  }
  .col-lg-pull-9 {
    right: 75%;
  }
  .col-lg-pull-8 {
    right: 66.66666667%;
  }
  .col-lg-pull-7 {
    right: 58.33333333%;
  }
  .col-lg-pull-6 {
    right: 50%;
  }
  .col-lg-pull-5 {
    right: 41.66666667%;
  }
  .col-lg-pull-4 {
    right: 33.33333333%;
  }
  .col-lg-pull-3 {
    right: 25%;
  }
  .col-lg-pull-2 {
    right: 16.66666667%;
  }
  .col-lg-pull-1 {
    right: 8.33333333%;
  }
  .col-lg-pull-0 {
    right: auto;
  }
  .col-lg-push-12 {
    left: 100%;
  }
  .col-lg-push-11 {
    left: 91.66666667%;
  }
  .col-lg-push-10 {
    left: 83.33333333%;
  }
  .col-lg-push-9 {
    left: 75%;
  }
  .col-lg-push-8 {
    left: 66.66666667%;
  }
  .col-lg-push-7 {
    left: 58.33333333%;
  }
  .col-lg-push-6 {
    left: 50%;
  }
  .col-lg-push-5 {
    left: 41.66666667%;
  }
  .col-lg-push-4 {
    left: 33.33333333%;
  }
  .col-lg-push-3 {
    left: 25%;
  }
  .col-lg-push-2 {
    left: 16.66666667%;
  }
  .col-lg-push-1 {
    left: 8.33333333%;
  }
  .col-lg-push-0 {
    left: auto;
  }
  .col-lg-offset-12 {
    margin-left: 100%;
  }
  .col-lg-offset-11 {
    margin-left: 91.66666667%;
  }
  .col-lg-offset-10 {
    margin-left: 83.33333333%;
  }
  .col-lg-offset-9 {
    margin-left: 75%;
  }
  .col-lg-offset-8 {
    margin-left: 66.66666667%;
  }
  .col-lg-offset-7 {
    margin-left: 58.33333333%;
  }
  .col-lg-offset-6 {
    margin-left: 50%;
  }
  .col-lg-offset-5 {
    margin-left: 41.66666667%;
  }
  .col-lg-offset-4 {
    margin-left: 33.33333333%;
  }
  .col-lg-offset-3 {
    margin-left: 25%;
  }
  .col-lg-offset-2 {
    margin-left: 16.66666667%;
  }
  .col-lg-offset-1 {
    margin-left: 8.33333333%;
  }
  .col-lg-offset-0 {
    margin-left: 0%;
  }
}
table {
  background-color: transparent;
}
caption {
  padding-top: 8px;
  padding-bottom: 8px;
  color: #777777;
  text-align: left;
}
th {
  text-align: left;
}
.table {
  width: 100%;
  max-width: 100%;
  margin-bottom: 18px;
}
.table > thead > tr > th,
.table > tbody > tr > th,
.table > tfoot > tr > th,
.table > thead > tr > td,
.table > tbody > tr > td,
.table > tfoot > tr > td {
  padding: 8px;
  line-height: 1.42857143;
  vertical-align: top;
  border-top: 1px solid #ddd;
}
.table > thead > tr > th {
  vertical-align: bottom;
  border-bottom: 2px solid #ddd;
}
.table > caption + thead > tr:first-child > th,
.table > colgroup + thead > tr:first-child > th,
.table > thead:first-child > tr:first-child > th,
.table > caption + thead > tr:first-child > td,
.table > colgroup + thead > tr:first-child > td,
.table > thead:first-child > tr:first-child > td {
  border-top: 0;
}
.table > tbody + tbody {
  border-top: 2px solid #ddd;
}
.table .table {
  background-color: #fff;
}
.table-condensed > thead > tr > th,
.table-condensed > tbody > tr > th,
.table-condensed > tfoot > tr > th,
.table-condensed > thead > tr > td,
.table-condensed > tbody > tr > td,
.table-condensed > tfoot > tr > td {
  padding: 5px;
}
.table-bordered {
  border: 1px solid #ddd;
}
.table-bordered > thead > tr > th,
.table-bordered > tbody > tr > th,
.table-bordered > tfoot > tr > th,
.table-bordered > thead > tr > td,
.table-bordered > tbody > tr > td,
.table-bordered > tfoot > tr > td {
  border: 1px solid #ddd;
}
.table-bordered > thead > tr > th,
.table-bordered > thead > tr > td {
  border-bottom-width: 2px;
}
.table-striped > tbody > tr:nth-of-type(odd) {
  background-color: #f9f9f9;
}
.table-hover > tbody > tr:hover {
  background-color: #f5f5f5;
}
table col[class*="col-"] {
  position: static;
  float: none;
  display: table-column;
}
table td[class*="col-"],
table th[class*="col-"] {
  position: static;
  float: none;
  display: table-cell;
}
.table > thead > tr > td.active,
.table > tbody > tr > td.active,
.table > tfoot > tr > td.active,
.table > thead > tr > th.active,
.table > tbody > tr > th.active,
.table > tfoot > tr > th.active,
.table > thead > tr.active > td,
.table > tbody > tr.active > td,
.table > tfoot > tr.active > td,
.table > thead > tr.active > th,
.table > tbody > tr.active > th,
.table > tfoot > tr.active > th {
  background-color: #f5f5f5;
}
.table-hover > tbody > tr > td.active:hover,
.table-hover > tbody > tr > th.active:hover,
.table-hover > tbody > tr.active:hover > td,
.table-hover > tbody > tr:hover > .active,
.table-hover > tbody > tr.active:hover > th {
  background-color: #e8e8e8;
}
.table > thead > tr > td.success,
.table > tbody > tr > td.success,
.table > tfoot > tr > td.success,
.table > thead > tr > th.success,
.table > tbody > tr > th.success,
.table > tfoot > tr > th.success,
.table > thead > tr.success > td,
.table > tbody > tr.success > td,
.table > tfoot > tr.success > td,
.table > thead > tr.success > th,
.table > tbody > tr.success > th,
.table > tfoot > tr.success > th {
  background-color: #dff0d8;
}
.table-hover > tbody > tr > td.success:hover,
.table-hover > tbody > tr > th.success:hover,
.table-hover > tbody > tr.success:hover > td,
.table-hover > tbody > tr:hover > .success,
.table-hover > tbody > tr.success:hover > th {
  background-color: #d0e9c6;
}
.table > thead > tr > td.info,
.table > tbody > tr > td.info,
.table > tfoot > tr > td.info,
.table > thead > tr > th.info,
.table > tbody > tr > th.info,
.table > tfoot > tr > th.info,
.table > thead > tr.info > td,
.table > tbody > tr.info > td,
.table > tfoot > tr.info > td,
.table > thead > tr.info > th,
.table > tbody > tr.info > th,
.table > tfoot > tr.info > th {
  background-color: #d9edf7;
}
.table-hover > tbody > tr > td.info:hover,
.table-hover > tbody > tr > th.info:hover,
.table-hover > tbody > tr.info:hover > td,
.table-hover > tbody > tr:hover > .info,
.table-hover > tbody > tr.info:hover > th {
  background-color: #c4e3f3;
}
.table > thead > tr > td.warning,
.table > tbody > tr > td.warning,
.table > tfoot > tr > td.warning,
.table > thead > tr > th.warning,
.table > tbody > tr > th.warning,
.table > tfoot > tr > th.warning,
.table > thead > tr.warning > td,
.table > tbody > tr.warning > td,
.table > tfoot > tr.warning > td,
.table > thead > tr.warning > th,
.table > tbody > tr.warning > th,
.table > tfoot > tr.warning > th {
  background-color: #fcf8e3;
}
.table-hover > tbody > tr > td.warning:hover,
.table-hover > tbody > tr > th.warning:hover,
.table-hover > tbody > tr.warning:hover > td,
.table-hover > tbody > tr:hover > .warning,
.table-hover > tbody > tr.warning:hover > th {
  background-color: #faf2cc;
}
.table > thead > tr > td.danger,
.table > tbody > tr > td.danger,
.table > tfoot > tr > td.danger,
.table > thead > tr > th.danger,
.table > tbody > tr > th.danger,
.table > tfoot > tr > th.danger,
.table > thead > tr.danger > td,
.table > tbody > tr.danger > td,
.table > tfoot > tr.danger > td,
.table > thead > tr.danger > th,
.table > tbody > tr.danger > th,
.table > tfoot > tr.danger > th {
  background-color: #f2dede;
}
.table-hover > tbody > tr > td.danger:hover,
.table-hover > tbody > tr > th.danger:hover,
.table-hover > tbody > tr.danger:hover > td,
.table-hover > tbody > tr:hover > .danger,
.table-hover > tbody > tr.danger:hover > th {
  background-color: #ebcccc;
}
.table-responsive {
  overflow-x: auto;
  min-height: 0.01%;
}
@media screen and (max-width: 767px) {
  .table-responsive {
    width: 100%;
    margin-bottom: 13.5px;
    overflow-y: hidden;
    -ms-overflow-style: -ms-autohiding-scrollbar;
    border: 1px solid #ddd;
  }
  .table-responsive > .table {
    margin-bottom: 0;
  }
  .table-responsive > .table > thead > tr > th,
  .table-responsive > .table > tbody > tr > th,
  .table-responsive > .table > tfoot > tr > th,
  .table-responsive > .table > thead > tr > td,
  .table-responsive > .table > tbody > tr > td,
  .table-responsive > .table > tfoot > tr > td {
    white-space: nowrap;
  }
  .table-responsive > .table-bordered {
    border: 0;
  }
  .table-responsive > .table-bordered > thead > tr > th:first-child,
  .table-responsive > .table-bordered > tbody > tr > th:first-child,
  .table-responsive > .table-bordered > tfoot > tr > th:first-child,
  .table-responsive > .table-bordered > thead > tr > td:first-child,
  .table-responsive > .table-bordered > tbody > tr > td:first-child,
  .table-responsive > .table-bordered > tfoot > tr > td:first-child {
    border-left: 0;
  }
  .table-responsive > .table-bordered > thead > tr > th:last-child,
  .table-responsive > .table-bordered > tbody > tr > th:last-child,
  .table-responsive > .table-bordered > tfoot > tr > th:last-child,
  .table-responsive > .table-bordered > thead > tr > td:last-child,
  .table-responsive > .table-bordered > tbody > tr > td:last-child,
  .table-responsive > .table-bordered > tfoot > tr > td:last-child {
    border-right: 0;
  }
  .table-responsive > .table-bordered > tbody > tr:last-child > th,
  .table-responsive > .table-bordered > tfoot > tr:last-child > th,
  .table-responsive > .table-bordered > tbody > tr:last-child > td,
  .table-responsive > .table-bordered > tfoot > tr:last-child > td {
    border-bottom: 0;
  }
}
fieldset {
  padding: 0;
  margin: 0;
  border: 0;
  min-width: 0;
}
legend {
  display: block;
  width: 100%;
  padding: 0;
  margin-bottom: 18px;
  font-size: 19.5px;
  line-height: inherit;
  color: #333333;
  border: 0;
  border-bottom: 1px solid #e5e5e5;
}
label {
  display: inline-block;
  max-width: 100%;
  margin-bottom: 5px;
  font-weight: bold;
}
input[type="search"] {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
input[type="radio"],
input[type="checkbox"] {
  margin: 4px 0 0;
  margin-top: 1px \9;
  line-height: normal;
}
input[type="file"] {
  display: block;
}
input[type="range"] {
  display: block;
  width: 100%;
}
select[multiple],
select[size] {
  height: auto;
}
input[type="file"]:focus,
input[type="radio"]:focus,
input[type="checkbox"]:focus {
  outline: thin dotted;
  outline: 5px auto -webkit-focus-ring-color;
  outline-offset: -2px;
}
output {
  display: block;
  padding-top: 7px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
}
.form-control {
  display: block;
  width: 100%;
  height: 32px;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
  background-color: #fff;
  background-image: none;
  border: 1px solid #ccc;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  -webkit-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  -o-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
}
.form-control:focus {
  border-color: #66afe9;
  outline: 0;
  -webkit-box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
  box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
}
.form-control::-moz-placeholder {
  color: #999;
  opacity: 1;
}
.form-control:-ms-input-placeholder {
  color: #999;
}
.form-control::-webkit-input-placeholder {
  color: #999;
}
.form-control::-ms-expand {
  border: 0;
  background-color: transparent;
}
.form-control[disabled],
.form-control[readonly],
fieldset[disabled] .form-control {
  background-color: #eeeeee;
  opacity: 1;
}
.form-control[disabled],
fieldset[disabled] .form-control {
  cursor: not-allowed;
}
textarea.form-control {
  height: auto;
}
input[type="search"] {
  -webkit-appearance: none;
}
@media screen and (-webkit-min-device-pixel-ratio: 0) {
  input[type="date"].form-control,
  input[type="time"].form-control,
  input[type="datetime-local"].form-control,
  input[type="month"].form-control {
    line-height: 32px;
  }
  input[type="date"].input-sm,
  input[type="time"].input-sm,
  input[type="datetime-local"].input-sm,
  input[type="month"].input-sm,
  .input-group-sm input[type="date"],
  .input-group-sm input[type="time"],
  .input-group-sm input[type="datetime-local"],
  .input-group-sm input[type="month"] {
    line-height: 30px;
  }
  input[type="date"].input-lg,
  input[type="time"].input-lg,
  input[type="datetime-local"].input-lg,
  input[type="month"].input-lg,
  .input-group-lg input[type="date"],
  .input-group-lg input[type="time"],
  .input-group-lg input[type="datetime-local"],
  .input-group-lg input[type="month"] {
    line-height: 45px;
  }
}
.form-group {
  margin-bottom: 15px;
}
.radio,
.checkbox {
  position: relative;
  display: block;
  margin-top: 10px;
  margin-bottom: 10px;
}
.radio label,
.checkbox label {
  min-height: 18px;
  padding-left: 20px;
  margin-bottom: 0;
  font-weight: normal;
  cursor: pointer;
}
.radio input[type="radio"],
.radio-inline input[type="radio"],
.checkbox input[type="checkbox"],
.checkbox-inline input[type="checkbox"] {
  position: absolute;
  margin-left: -20px;
  margin-top: 4px \9;
}
.radio + .radio,
.checkbox + .checkbox {
  margin-top: -5px;
}
.radio-inline,
.checkbox-inline {
  position: relative;
  display: inline-block;
  padding-left: 20px;
  margin-bottom: 0;
  vertical-align: middle;
  font-weight: normal;
  cursor: pointer;
}
.radio-inline + .radio-inline,
.checkbox-inline + .checkbox-inline {
  margin-top: 0;
  margin-left: 10px;
}
input[type="radio"][disabled],
input[type="checkbox"][disabled],
input[type="radio"].disabled,
input[type="checkbox"].disabled,
fieldset[disabled] input[type="radio"],
fieldset[disabled] input[type="checkbox"] {
  cursor: not-allowed;
}
.radio-inline.disabled,
.checkbox-inline.disabled,
fieldset[disabled] .radio-inline,
fieldset[disabled] .checkbox-inline {
  cursor: not-allowed;
}
.radio.disabled label,
.checkbox.disabled label,
fieldset[disabled] .radio label,
fieldset[disabled] .checkbox label {
  cursor: not-allowed;
}
.form-control-static {
  padding-top: 7px;
  padding-bottom: 7px;
  margin-bottom: 0;
  min-height: 31px;
}
.form-control-static.input-lg,
.form-control-static.input-sm {
  padding-left: 0;
  padding-right: 0;
}
.input-sm {
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
select.input-sm {
  height: 30px;
  line-height: 30px;
}
textarea.input-sm,
select[multiple].input-sm {
  height: auto;
}
.form-group-sm .form-control {
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
.form-group-sm select.form-control {
  height: 30px;
  line-height: 30px;
}
.form-group-sm textarea.form-control,
.form-group-sm select[multiple].form-control {
  height: auto;
}
.form-group-sm .form-control-static {
  height: 30px;
  min-height: 30px;
  padding: 6px 10px;
  font-size: 12px;
  line-height: 1.5;
}
.input-lg {
  height: 45px;
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
select.input-lg {
  height: 45px;
  line-height: 45px;
}
textarea.input-lg,
select[multiple].input-lg {
  height: auto;
}
.form-group-lg .form-control {
  height: 45px;
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
.form-group-lg select.form-control {
  height: 45px;
  line-height: 45px;
}
.form-group-lg textarea.form-control,
.form-group-lg select[multiple].form-control {
  height: auto;
}
.form-group-lg .form-control-static {
  height: 45px;
  min-height: 35px;
  padding: 11px 16px;
  font-size: 17px;
  line-height: 1.3333333;
}
.has-feedback {
  position: relative;
}
.has-feedback .form-control {
  padding-right: 40px;
}
.form-control-feedback {
  position: absolute;
  top: 0;
  right: 0;
  z-index: 2;
  display: block;
  width: 32px;
  height: 32px;
  line-height: 32px;
  text-align: center;
  pointer-events: none;
}
.input-lg + .form-control-feedback,
.input-group-lg + .form-control-feedback,
.form-group-lg .form-control + .form-control-feedback {
  width: 45px;
  height: 45px;
  line-height: 45px;
}
.input-sm + .form-control-feedback,
.input-group-sm + .form-control-feedback,
.form-group-sm .form-control + .form-control-feedback {
  width: 30px;
  height: 30px;
  line-height: 30px;
}
.has-success .help-block,
.has-success .control-label,
.has-success .radio,
.has-success .checkbox,
.has-success .radio-inline,
.has-success .checkbox-inline,
.has-success.radio label,
.has-success.checkbox label,
.has-success.radio-inline label,
.has-success.checkbox-inline label {
  color: #3c763d;
}
.has-success .form-control {
  border-color: #3c763d;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
}
.has-success .form-control:focus {
  border-color: #2b542c;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #67b168;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #67b168;
}
.has-success .input-group-addon {
  color: #3c763d;
  border-color: #3c763d;
  background-color: #dff0d8;
}
.has-success .form-control-feedback {
  color: #3c763d;
}
.has-warning .help-block,
.has-warning .control-label,
.has-warning .radio,
.has-warning .checkbox,
.has-warning .radio-inline,
.has-warning .checkbox-inline,
.has-warning.radio label,
.has-warning.checkbox label,
.has-warning.radio-inline label,
.has-warning.checkbox-inline label {
  color: #8a6d3b;
}
.has-warning .form-control {
  border-color: #8a6d3b;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
}
.has-warning .form-control:focus {
  border-color: #66512c;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #c0a16b;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #c0a16b;
}
.has-warning .input-group-addon {
  color: #8a6d3b;
  border-color: #8a6d3b;
  background-color: #fcf8e3;
}
.has-warning .form-control-feedback {
  color: #8a6d3b;
}
.has-error .help-block,
.has-error .control-label,
.has-error .radio,
.has-error .checkbox,
.has-error .radio-inline,
.has-error .checkbox-inline,
.has-error.radio label,
.has-error.checkbox label,
.has-error.radio-inline label,
.has-error.checkbox-inline label {
  color: #a94442;
}
.has-error .form-control {
  border-color: #a94442;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
}
.has-error .form-control:focus {
  border-color: #843534;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #ce8483;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #ce8483;
}
.has-error .input-group-addon {
  color: #a94442;
  border-color: #a94442;
  background-color: #f2dede;
}
.has-error .form-control-feedback {
  color: #a94442;
}
.has-feedback label ~ .form-control-feedback {
  top: 23px;
}
.has-feedback label.sr-only ~ .form-control-feedback {
  top: 0;
}
.help-block {
  display: block;
  margin-top: 5px;
  margin-bottom: 10px;
  color: #404040;
}
@media (min-width: 768px) {
  .form-inline .form-group {
    display: inline-block;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .form-inline .form-control {
    display: inline-block;
    width: auto;
    vertical-align: middle;
  }
  .form-inline .form-control-static {
    display: inline-block;
  }
  .form-inline .input-group {
    display: inline-table;
    vertical-align: middle;
  }
  .form-inline .input-group .input-group-addon,
  .form-inline .input-group .input-group-btn,
  .form-inline .input-group .form-control {
    width: auto;
  }
  .form-inline .input-group > .form-control {
    width: 100%;
  }
  .form-inline .control-label {
    margin-bottom: 0;
    vertical-align: middle;
  }
  .form-inline .radio,
  .form-inline .checkbox {
    display: inline-block;
    margin-top: 0;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .form-inline .radio label,
  .form-inline .checkbox label {
    padding-left: 0;
  }
  .form-inline .radio input[type="radio"],
  .form-inline .checkbox input[type="checkbox"] {
    position: relative;
    margin-left: 0;
  }
  .form-inline .has-feedback .form-control-feedback {
    top: 0;
  }
}
.form-horizontal .radio,
.form-horizontal .checkbox,
.form-horizontal .radio-inline,
.form-horizontal .checkbox-inline {
  margin-top: 0;
  margin-bottom: 0;
  padding-top: 7px;
}
.form-horizontal .radio,
.form-horizontal .checkbox {
  min-height: 25px;
}
.form-horizontal .form-group {
  margin-left: 0px;
  margin-right: 0px;
}
@media (min-width: 768px) {
  .form-horizontal .control-label {
    text-align: right;
    margin-bottom: 0;
    padding-top: 7px;
  }
}
.form-horizontal .has-feedback .form-control-feedback {
  right: 0px;
}
@media (min-width: 768px) {
  .form-horizontal .form-group-lg .control-label {
    padding-top: 11px;
    font-size: 17px;
  }
}
@media (min-width: 768px) {
  .form-horizontal .form-group-sm .control-label {
    padding-top: 6px;
    font-size: 12px;
  }
}
.btn {
  display: inline-block;
  margin-bottom: 0;
  font-weight: normal;
  text-align: center;
  vertical-align: middle;
  touch-action: manipulation;
  cursor: pointer;
  background-image: none;
  border: 1px solid transparent;
  white-space: nowrap;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  border-radius: 2px;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}
.btn:focus,
.btn:active:focus,
.btn.active:focus,
.btn.focus,
.btn:active.focus,
.btn.active.focus {
  outline: thin dotted;
  outline: 5px auto -webkit-focus-ring-color;
  outline-offset: -2px;
}
.btn:hover,
.btn:focus,
.btn.focus {
  color: #333;
  text-decoration: none;
}
.btn:active,
.btn.active {
  outline: 0;
  background-image: none;
  -webkit-box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
  box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
}
.btn.disabled,
.btn[disabled],
fieldset[disabled] .btn {
  cursor: not-allowed;
  opacity: 0.65;
  filter: alpha(opacity=65);
  -webkit-box-shadow: none;
  box-shadow: none;
}
a.btn.disabled,
fieldset[disabled] a.btn {
  pointer-events: none;
}
.btn-default {
  color: #333;
  background-color: #fff;
  border-color: #ccc;
}
.btn-default:focus,
.btn-default.focus {
  color: #333;
  background-color: #e6e6e6;
  border-color: #8c8c8c;
}
.btn-default:hover {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.btn-default:active,
.btn-default.active,
.open > .dropdown-toggle.btn-default {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.btn-default:active:hover,
.btn-default.active:hover,
.open > .dropdown-toggle.btn-default:hover,
.btn-default:active:focus,
.btn-default.active:focus,
.open > .dropdown-toggle.btn-default:focus,
.btn-default:active.focus,
.btn-default.active.focus,
.open > .dropdown-toggle.btn-default.focus {
  color: #333;
  background-color: #d4d4d4;
  border-color: #8c8c8c;
}
.btn-default:active,
.btn-default.active,
.open > .dropdown-toggle.btn-default {
  background-image: none;
}
.btn-default.disabled:hover,
.btn-default[disabled]:hover,
fieldset[disabled] .btn-default:hover,
.btn-default.disabled:focus,
.btn-default[disabled]:focus,
fieldset[disabled] .btn-default:focus,
.btn-default.disabled.focus,
.btn-default[disabled].focus,
fieldset[disabled] .btn-default.focus {
  background-color: #fff;
  border-color: #ccc;
}
.btn-default .badge {
  color: #fff;
  background-color: #333;
}
.btn-primary {
  color: #fff;
  background-color: #337ab7;
  border-color: #2e6da4;
}
.btn-primary:focus,
.btn-primary.focus {
  color: #fff;
  background-color: #286090;
  border-color: #122b40;
}
.btn-primary:hover {
  color: #fff;
  background-color: #286090;
  border-color: #204d74;
}
.btn-primary:active,
.btn-primary.active,
.open > .dropdown-toggle.btn-primary {
  color: #fff;
  background-color: #286090;
  border-color: #204d74;
}
.btn-primary:active:hover,
.btn-primary.active:hover,
.open > .dropdown-toggle.btn-primary:hover,
.btn-primary:active:focus,
.btn-primary.active:focus,
.open > .dropdown-toggle.btn-primary:focus,
.btn-primary:active.focus,
.btn-primary.active.focus,
.open > .dropdown-toggle.btn-primary.focus {
  color: #fff;
  background-color: #204d74;
  border-color: #122b40;
}
.btn-primary:active,
.btn-primary.active,
.open > .dropdown-toggle.btn-primary {
  background-image: none;
}
.btn-primary.disabled:hover,
.btn-primary[disabled]:hover,
fieldset[disabled] .btn-primary:hover,
.btn-primary.disabled:focus,
.btn-primary[disabled]:focus,
fieldset[disabled] .btn-primary:focus,
.btn-primary.disabled.focus,
.btn-primary[disabled].focus,
fieldset[disabled] .btn-primary.focus {
  background-color: #337ab7;
  border-color: #2e6da4;
}
.btn-primary .badge {
  color: #337ab7;
  background-color: #fff;
}
.btn-success {
  color: #fff;
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.btn-success:focus,
.btn-success.focus {
  color: #fff;
  background-color: #449d44;
  border-color: #255625;
}
.btn-success:hover {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.btn-success:active,
.btn-success.active,
.open > .dropdown-toggle.btn-success {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.btn-success:active:hover,
.btn-success.active:hover,
.open > .dropdown-toggle.btn-success:hover,
.btn-success:active:focus,
.btn-success.active:focus,
.open > .dropdown-toggle.btn-success:focus,
.btn-success:active.focus,
.btn-success.active.focus,
.open > .dropdown-toggle.btn-success.focus {
  color: #fff;
  background-color: #398439;
  border-color: #255625;
}
.btn-success:active,
.btn-success.active,
.open > .dropdown-toggle.btn-success {
  background-image: none;
}
.btn-success.disabled:hover,
.btn-success[disabled]:hover,
fieldset[disabled] .btn-success:hover,
.btn-success.disabled:focus,
.btn-success[disabled]:focus,
fieldset[disabled] .btn-success:focus,
.btn-success.disabled.focus,
.btn-success[disabled].focus,
fieldset[disabled] .btn-success.focus {
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.btn-success .badge {
  color: #5cb85c;
  background-color: #fff;
}
.btn-info {
  color: #fff;
  background-color: #5bc0de;
  border-color: #46b8da;
}
.btn-info:focus,
.btn-info.focus {
  color: #fff;
  background-color: #31b0d5;
  border-color: #1b6d85;
}
.btn-info:hover {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.btn-info:active,
.btn-info.active,
.open > .dropdown-toggle.btn-info {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.btn-info:active:hover,
.btn-info.active:hover,
.open > .dropdown-toggle.btn-info:hover,
.btn-info:active:focus,
.btn-info.active:focus,
.open > .dropdown-toggle.btn-info:focus,
.btn-info:active.focus,
.btn-info.active.focus,
.open > .dropdown-toggle.btn-info.focus {
  color: #fff;
  background-color: #269abc;
  border-color: #1b6d85;
}
.btn-info:active,
.btn-info.active,
.open > .dropdown-toggle.btn-info {
  background-image: none;
}
.btn-info.disabled:hover,
.btn-info[disabled]:hover,
fieldset[disabled] .btn-info:hover,
.btn-info.disabled:focus,
.btn-info[disabled]:focus,
fieldset[disabled] .btn-info:focus,
.btn-info.disabled.focus,
.btn-info[disabled].focus,
fieldset[disabled] .btn-info.focus {
  background-color: #5bc0de;
  border-color: #46b8da;
}
.btn-info .badge {
  color: #5bc0de;
  background-color: #fff;
}
.btn-warning {
  color: #fff;
  background-color: #f0ad4e;
  border-color: #eea236;
}
.btn-warning:focus,
.btn-warning.focus {
  color: #fff;
  background-color: #ec971f;
  border-color: #985f0d;
}
.btn-warning:hover {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.btn-warning:active,
.btn-warning.active,
.open > .dropdown-toggle.btn-warning {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.btn-warning:active:hover,
.btn-warning.active:hover,
.open > .dropdown-toggle.btn-warning:hover,
.btn-warning:active:focus,
.btn-warning.active:focus,
.open > .dropdown-toggle.btn-warning:focus,
.btn-warning:active.focus,
.btn-warning.active.focus,
.open > .dropdown-toggle.btn-warning.focus {
  color: #fff;
  background-color: #d58512;
  border-color: #985f0d;
}
.btn-warning:active,
.btn-warning.active,
.open > .dropdown-toggle.btn-warning {
  background-image: none;
}
.btn-warning.disabled:hover,
.btn-warning[disabled]:hover,
fieldset[disabled] .btn-warning:hover,
.btn-warning.disabled:focus,
.btn-warning[disabled]:focus,
fieldset[disabled] .btn-warning:focus,
.btn-warning.disabled.focus,
.btn-warning[disabled].focus,
fieldset[disabled] .btn-warning.focus {
  background-color: #f0ad4e;
  border-color: #eea236;
}
.btn-warning .badge {
  color: #f0ad4e;
  background-color: #fff;
}
.btn-danger {
  color: #fff;
  background-color: #d9534f;
  border-color: #d43f3a;
}
.btn-danger:focus,
.btn-danger.focus {
  color: #fff;
  background-color: #c9302c;
  border-color: #761c19;
}
.btn-danger:hover {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.btn-danger:active,
.btn-danger.active,
.open > .dropdown-toggle.btn-danger {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.btn-danger:active:hover,
.btn-danger.active:hover,
.open > .dropdown-toggle.btn-danger:hover,
.btn-danger:active:focus,
.btn-danger.active:focus,
.open > .dropdown-toggle.btn-danger:focus,
.btn-danger:active.focus,
.btn-danger.active.focus,
.open > .dropdown-toggle.btn-danger.focus {
  color: #fff;
  background-color: #ac2925;
  border-color: #761c19;
}
.btn-danger:active,
.btn-danger.active,
.open > .dropdown-toggle.btn-danger {
  background-image: none;
}
.btn-danger.disabled:hover,
.btn-danger[disabled]:hover,
fieldset[disabled] .btn-danger:hover,
.btn-danger.disabled:focus,
.btn-danger[disabled]:focus,
fieldset[disabled] .btn-danger:focus,
.btn-danger.disabled.focus,
.btn-danger[disabled].focus,
fieldset[disabled] .btn-danger.focus {
  background-color: #d9534f;
  border-color: #d43f3a;
}
.btn-danger .badge {
  color: #d9534f;
  background-color: #fff;
}
.btn-link {
  color: #337ab7;
  font-weight: normal;
  border-radius: 0;
}
.btn-link,
.btn-link:active,
.btn-link.active,
.btn-link[disabled],
fieldset[disabled] .btn-link {
  background-color: transparent;
  -webkit-box-shadow: none;
  box-shadow: none;
}
.btn-link,
.btn-link:hover,
.btn-link:focus,
.btn-link:active {
  border-color: transparent;
}
.btn-link:hover,
.btn-link:focus {
  color: #23527c;
  text-decoration: underline;
  background-color: transparent;
}
.btn-link[disabled]:hover,
fieldset[disabled] .btn-link:hover,
.btn-link[disabled]:focus,
fieldset[disabled] .btn-link:focus {
  color: #777777;
  text-decoration: none;
}
.btn-lg,
.btn-group-lg > .btn {
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
.btn-sm,
.btn-group-sm > .btn {
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
.btn-xs,
.btn-group-xs > .btn {
  padding: 1px 5px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
.btn-block {
  display: block;
  width: 100%;
}
.btn-block + .btn-block {
  margin-top: 5px;
}
input[type="submit"].btn-block,
input[type="reset"].btn-block,
input[type="button"].btn-block {
  width: 100%;
}
.fade {
  opacity: 0;
  -webkit-transition: opacity 0.15s linear;
  -o-transition: opacity 0.15s linear;
  transition: opacity 0.15s linear;
}
.fade.in {
  opacity: 1;
}
.collapse {
  display: none;
}
.collapse.in {
  display: block;
}
tr.collapse.in {
  display: table-row;
}
tbody.collapse.in {
  display: table-row-group;
}
.collapsing {
  position: relative;
  height: 0;
  overflow: hidden;
  -webkit-transition-property: height, visibility;
  transition-property: height, visibility;
  -webkit-transition-duration: 0.35s;
  transition-duration: 0.35s;
  -webkit-transition-timing-function: ease;
  transition-timing-function: ease;
}
.caret {
  display: inline-block;
  width: 0;
  height: 0;
  margin-left: 2px;
  vertical-align: middle;
  border-top: 4px dashed;
  border-top: 4px solid \9;
  border-right: 4px solid transparent;
  border-left: 4px solid transparent;
}
.dropup,
.dropdown {
  position: relative;
}
.dropdown-toggle:focus {
  outline: 0;
}
.dropdown-menu {
  position: absolute;
  top: 100%;
  left: 0;
  z-index: 1000;
  display: none;
  float: left;
  min-width: 160px;
  padding: 5px 0;
  margin: 2px 0 0;
  list-style: none;
  font-size: 13px;
  text-align: left;
  background-color: #fff;
  border: 1px solid #ccc;
  border: 1px solid rgba(0, 0, 0, 0.15);
  border-radius: 2px;
  -webkit-box-shadow: 0 6px 12px rgba(0, 0, 0, 0.175);
  box-shadow: 0 6px 12px rgba(0, 0, 0, 0.175);
  background-clip: padding-box;
}
.dropdown-menu.pull-right {
  right: 0;
  left: auto;
}
.dropdown-menu .divider {
  height: 1px;
  margin: 8px 0;
  overflow: hidden;
  background-color: #e5e5e5;
}
.dropdown-menu > li > a {
  display: block;
  padding: 3px 20px;
  clear: both;
  font-weight: normal;
  line-height: 1.42857143;
  color: #333333;
  white-space: nowrap;
}
.dropdown-menu > li > a:hover,
.dropdown-menu > li > a:focus {
  text-decoration: none;
  color: #262626;
  background-color: #f5f5f5;
}
.dropdown-menu > .active > a,
.dropdown-menu > .active > a:hover,
.dropdown-menu > .active > a:focus {
  color: #fff;
  text-decoration: none;
  outline: 0;
  background-color: #337ab7;
}
.dropdown-menu > .disabled > a,
.dropdown-menu > .disabled > a:hover,
.dropdown-menu > .disabled > a:focus {
  color: #777777;
}
.dropdown-menu > .disabled > a:hover,
.dropdown-menu > .disabled > a:focus {
  text-decoration: none;
  background-color: transparent;
  background-image: none;
  filter: progid:DXImageTransform.Microsoft.gradient(enabled = false);
  cursor: not-allowed;
}
.open > .dropdown-menu {
  display: block;
}
.open > a {
  outline: 0;
}
.dropdown-menu-right {
  left: auto;
  right: 0;
}
.dropdown-menu-left {
  left: 0;
  right: auto;
}
.dropdown-header {
  display: block;
  padding: 3px 20px;
  font-size: 12px;
  line-height: 1.42857143;
  color: #777777;
  white-space: nowrap;
}
.dropdown-backdrop {
  position: fixed;
  left: 0;
  right: 0;
  bottom: 0;
  top: 0;
  z-index: 990;
}
.pull-right > .dropdown-menu {
  right: 0;
  left: auto;
}
.dropup .caret,
.navbar-fixed-bottom .dropdown .caret {
  border-top: 0;
  border-bottom: 4px dashed;
  border-bottom: 4px solid \9;
  content: "";
}
.dropup .dropdown-menu,
.navbar-fixed-bottom .dropdown .dropdown-menu {
  top: auto;
  bottom: 100%;
  margin-bottom: 2px;
}
@media (min-width: 541px) {
  .navbar-right .dropdown-menu {
    left: auto;
    right: 0;
  }
  .navbar-right .dropdown-menu-left {
    left: 0;
    right: auto;
  }
}
.btn-group,
.btn-group-vertical {
  position: relative;
  display: inline-block;
  vertical-align: middle;
}
.btn-group > .btn,
.btn-group-vertical > .btn {
  position: relative;
  float: left;
}
.btn-group > .btn:hover,
.btn-group-vertical > .btn:hover,
.btn-group > .btn:focus,
.btn-group-vertical > .btn:focus,
.btn-group > .btn:active,
.btn-group-vertical > .btn:active,
.btn-group > .btn.active,
.btn-group-vertical > .btn.active {
  z-index: 2;
}
.btn-group .btn + .btn,
.btn-group .btn + .btn-group,
.btn-group .btn-group + .btn,
.btn-group .btn-group + .btn-group {
  margin-left: -1px;
}
.btn-toolbar {
  margin-left: -5px;
}
.btn-toolbar .btn,
.btn-toolbar .btn-group,
.btn-toolbar .input-group {
  float: left;
}
.btn-toolbar > .btn,
.btn-toolbar > .btn-group,
.btn-toolbar > .input-group {
  margin-left: 5px;
}
.btn-group > .btn:not(:first-child):not(:last-child):not(.dropdown-toggle) {
  border-radius: 0;
}
.btn-group > .btn:first-child {
  margin-left: 0;
}
.btn-group > .btn:first-child:not(:last-child):not(.dropdown-toggle) {
  border-bottom-right-radius: 0;
  border-top-right-radius: 0;
}
.btn-group > .btn:last-child:not(:first-child),
.btn-group > .dropdown-toggle:not(:first-child) {
  border-bottom-left-radius: 0;
  border-top-left-radius: 0;
}
.btn-group > .btn-group {
  float: left;
}
.btn-group > .btn-group:not(:first-child):not(:last-child) > .btn {
  border-radius: 0;
}
.btn-group > .btn-group:first-child:not(:last-child) > .btn:last-child,
.btn-group > .btn-group:first-child:not(:last-child) > .dropdown-toggle {
  border-bottom-right-radius: 0;
  border-top-right-radius: 0;
}
.btn-group > .btn-group:last-child:not(:first-child) > .btn:first-child {
  border-bottom-left-radius: 0;
  border-top-left-radius: 0;
}
.btn-group .dropdown-toggle:active,
.btn-group.open .dropdown-toggle {
  outline: 0;
}
.btn-group > .btn + .dropdown-toggle {
  padding-left: 8px;
  padding-right: 8px;
}
.btn-group > .btn-lg + .dropdown-toggle {
  padding-left: 12px;
  padding-right: 12px;
}
.btn-group.open .dropdown-toggle {
  -webkit-box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
  box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
}
.btn-group.open .dropdown-toggle.btn-link {
  -webkit-box-shadow: none;
  box-shadow: none;
}
.btn .caret {
  margin-left: 0;
}
.btn-lg .caret {
  border-width: 5px 5px 0;
  border-bottom-width: 0;
}
.dropup .btn-lg .caret {
  border-width: 0 5px 5px;
}
.btn-group-vertical > .btn,
.btn-group-vertical > .btn-group,
.btn-group-vertical > .btn-group > .btn {
  display: block;
  float: none;
  width: 100%;
  max-width: 100%;
}
.btn-group-vertical > .btn-group > .btn {
  float: none;
}
.btn-group-vertical > .btn + .btn,
.btn-group-vertical > .btn + .btn-group,
.btn-group-vertical > .btn-group + .btn,
.btn-group-vertical > .btn-group + .btn-group {
  margin-top: -1px;
  margin-left: 0;
}
.btn-group-vertical > .btn:not(:first-child):not(:last-child) {
  border-radius: 0;
}
.btn-group-vertical > .btn:first-child:not(:last-child) {
  border-top-right-radius: 2px;
  border-top-left-radius: 2px;
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.btn-group-vertical > .btn:last-child:not(:first-child) {
  border-top-right-radius: 0;
  border-top-left-radius: 0;
  border-bottom-right-radius: 2px;
  border-bottom-left-radius: 2px;
}
.btn-group-vertical > .btn-group:not(:first-child):not(:last-child) > .btn {
  border-radius: 0;
}
.btn-group-vertical > .btn-group:first-child:not(:last-child) > .btn:last-child,
.btn-group-vertical > .btn-group:first-child:not(:last-child) > .dropdown-toggle {
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.btn-group-vertical > .btn-group:last-child:not(:first-child) > .btn:first-child {
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.btn-group-justified {
  display: table;
  width: 100%;
  table-layout: fixed;
  border-collapse: separate;
}
.btn-group-justified > .btn,
.btn-group-justified > .btn-group {
  float: none;
  display: table-cell;
  width: 1%;
}
.btn-group-justified > .btn-group .btn {
  width: 100%;
}
.btn-group-justified > .btn-group .dropdown-menu {
  left: auto;
}
[data-toggle="buttons"] > .btn input[type="radio"],
[data-toggle="buttons"] > .btn-group > .btn input[type="radio"],
[data-toggle="buttons"] > .btn input[type="checkbox"],
[data-toggle="buttons"] > .btn-group > .btn input[type="checkbox"] {
  position: absolute;
  clip: rect(0, 0, 0, 0);
  pointer-events: none;
}
.input-group {
  position: relative;
  display: table;
  border-collapse: separate;
}
.input-group[class*="col-"] {
  float: none;
  padding-left: 0;
  padding-right: 0;
}
.input-group .form-control {
  position: relative;
  z-index: 2;
  float: left;
  width: 100%;
  margin-bottom: 0;
}
.input-group .form-control:focus {
  z-index: 3;
}
.input-group-lg > .form-control,
.input-group-lg > .input-group-addon,
.input-group-lg > .input-group-btn > .btn {
  height: 45px;
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
select.input-group-lg > .form-control,
select.input-group-lg > .input-group-addon,
select.input-group-lg > .input-group-btn > .btn {
  height: 45px;
  line-height: 45px;
}
textarea.input-group-lg > .form-control,
textarea.input-group-lg > .input-group-addon,
textarea.input-group-lg > .input-group-btn > .btn,
select[multiple].input-group-lg > .form-control,
select[multiple].input-group-lg > .input-group-addon,
select[multiple].input-group-lg > .input-group-btn > .btn {
  height: auto;
}
.input-group-sm > .form-control,
.input-group-sm > .input-group-addon,
.input-group-sm > .input-group-btn > .btn {
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
select.input-group-sm > .form-control,
select.input-group-sm > .input-group-addon,
select.input-group-sm > .input-group-btn > .btn {
  height: 30px;
  line-height: 30px;
}
textarea.input-group-sm > .form-control,
textarea.input-group-sm > .input-group-addon,
textarea.input-group-sm > .input-group-btn > .btn,
select[multiple].input-group-sm > .form-control,
select[multiple].input-group-sm > .input-group-addon,
select[multiple].input-group-sm > .input-group-btn > .btn {
  height: auto;
}
.input-group-addon,
.input-group-btn,
.input-group .form-control {
  display: table-cell;
}
.input-group-addon:not(:first-child):not(:last-child),
.input-group-btn:not(:first-child):not(:last-child),
.input-group .form-control:not(:first-child):not(:last-child) {
  border-radius: 0;
}
.input-group-addon,
.input-group-btn {
  width: 1%;
  white-space: nowrap;
  vertical-align: middle;
}
.input-group-addon {
  padding: 6px 12px;
  font-size: 13px;
  font-weight: normal;
  line-height: 1;
  color: #555555;
  text-align: center;
  background-color: #eeeeee;
  border: 1px solid #ccc;
  border-radius: 2px;
}
.input-group-addon.input-sm {
  padding: 5px 10px;
  font-size: 12px;
  border-radius: 1px;
}
.input-group-addon.input-lg {
  padding: 10px 16px;
  font-size: 17px;
  border-radius: 3px;
}
.input-group-addon input[type="radio"],
.input-group-addon input[type="checkbox"] {
  margin-top: 0;
}
.input-group .form-control:first-child,
.input-group-addon:first-child,
.input-group-btn:first-child > .btn,
.input-group-btn:first-child > .btn-group > .btn,
.input-group-btn:first-child > .dropdown-toggle,
.input-group-btn:last-child > .btn:not(:last-child):not(.dropdown-toggle),
.input-group-btn:last-child > .btn-group:not(:last-child) > .btn {
  border-bottom-right-radius: 0;
  border-top-right-radius: 0;
}
.input-group-addon:first-child {
  border-right: 0;
}
.input-group .form-control:last-child,
.input-group-addon:last-child,
.input-group-btn:last-child > .btn,
.input-group-btn:last-child > .btn-group > .btn,
.input-group-btn:last-child > .dropdown-toggle,
.input-group-btn:first-child > .btn:not(:first-child),
.input-group-btn:first-child > .btn-group:not(:first-child) > .btn {
  border-bottom-left-radius: 0;
  border-top-left-radius: 0;
}
.input-group-addon:last-child {
  border-left: 0;
}
.input-group-btn {
  position: relative;
  font-size: 0;
  white-space: nowrap;
}
.input-group-btn > .btn {
  position: relative;
}
.input-group-btn > .btn + .btn {
  margin-left: -1px;
}
.input-group-btn > .btn:hover,
.input-group-btn > .btn:focus,
.input-group-btn > .btn:active {
  z-index: 2;
}
.input-group-btn:first-child > .btn,
.input-group-btn:first-child > .btn-group {
  margin-right: -1px;
}
.input-group-btn:last-child > .btn,
.input-group-btn:last-child > .btn-group {
  z-index: 2;
  margin-left: -1px;
}
.nav {
  margin-bottom: 0;
  padding-left: 0;
  list-style: none;
}
.nav > li {
  position: relative;
  display: block;
}
.nav > li > a {
  position: relative;
  display: block;
  padding: 10px 15px;
}
.nav > li > a:hover,
.nav > li > a:focus {
  text-decoration: none;
  background-color: #eeeeee;
}
.nav > li.disabled > a {
  color: #777777;
}
.nav > li.disabled > a:hover,
.nav > li.disabled > a:focus {
  color: #777777;
  text-decoration: none;
  background-color: transparent;
  cursor: not-allowed;
}
.nav .open > a,
.nav .open > a:hover,
.nav .open > a:focus {
  background-color: #eeeeee;
  border-color: #337ab7;
}
.nav .nav-divider {
  height: 1px;
  margin: 8px 0;
  overflow: hidden;
  background-color: #e5e5e5;
}
.nav > li > a > img {
  max-width: none;
}
.nav-tabs {
  border-bottom: 1px solid #ddd;
}
.nav-tabs > li {
  float: left;
  margin-bottom: -1px;
}
.nav-tabs > li > a {
  margin-right: 2px;
  line-height: 1.42857143;
  border: 1px solid transparent;
  border-radius: 2px 2px 0 0;
}
.nav-tabs > li > a:hover {
  border-color: #eeeeee #eeeeee #ddd;
}
.nav-tabs > li.active > a,
.nav-tabs > li.active > a:hover,
.nav-tabs > li.active > a:focus {
  color: #555555;
  background-color: #fff;
  border: 1px solid #ddd;
  border-bottom-color: transparent;
  cursor: default;
}
.nav-tabs.nav-justified {
  width: 100%;
  border-bottom: 0;
}
.nav-tabs.nav-justified > li {
  float: none;
}
.nav-tabs.nav-justified > li > a {
  text-align: center;
  margin-bottom: 5px;
}
.nav-tabs.nav-justified > .dropdown .dropdown-menu {
  top: auto;
  left: auto;
}
@media (min-width: 768px) {
  .nav-tabs.nav-justified > li {
    display: table-cell;
    width: 1%;
  }
  .nav-tabs.nav-justified > li > a {
    margin-bottom: 0;
  }
}
.nav-tabs.nav-justified > li > a {
  margin-right: 0;
  border-radius: 2px;
}
.nav-tabs.nav-justified > .active > a,
.nav-tabs.nav-justified > .active > a:hover,
.nav-tabs.nav-justified > .active > a:focus {
  border: 1px solid #ddd;
}
@media (min-width: 768px) {
  .nav-tabs.nav-justified > li > a {
    border-bottom: 1px solid #ddd;
    border-radius: 2px 2px 0 0;
  }
  .nav-tabs.nav-justified > .active > a,
  .nav-tabs.nav-justified > .active > a:hover,
  .nav-tabs.nav-justified > .active > a:focus {
    border-bottom-color: #fff;
  }
}
.nav-pills > li {
  float: left;
}
.nav-pills > li > a {
  border-radius: 2px;
}
.nav-pills > li + li {
  margin-left: 2px;
}
.nav-pills > li.active > a,
.nav-pills > li.active > a:hover,
.nav-pills > li.active > a:focus {
  color: #fff;
  background-color: #337ab7;
}
.nav-stacked > li {
  float: none;
}
.nav-stacked > li + li {
  margin-top: 2px;
  margin-left: 0;
}
.nav-justified {
  width: 100%;
}
.nav-justified > li {
  float: none;
}
.nav-justified > li > a {
  text-align: center;
  margin-bottom: 5px;
}
.nav-justified > .dropdown .dropdown-menu {
  top: auto;
  left: auto;
}
@media (min-width: 768px) {
  .nav-justified > li {
    display: table-cell;
    width: 1%;
  }
  .nav-justified > li > a {
    margin-bottom: 0;
  }
}
.nav-tabs-justified {
  border-bottom: 0;
}
.nav-tabs-justified > li > a {
  margin-right: 0;
  border-radius: 2px;
}
.nav-tabs-justified > .active > a,
.nav-tabs-justified > .active > a:hover,
.nav-tabs-justified > .active > a:focus {
  border: 1px solid #ddd;
}
@media (min-width: 768px) {
  .nav-tabs-justified > li > a {
    border-bottom: 1px solid #ddd;
    border-radius: 2px 2px 0 0;
  }
  .nav-tabs-justified > .active > a,
  .nav-tabs-justified > .active > a:hover,
  .nav-tabs-justified > .active > a:focus {
    border-bottom-color: #fff;
  }
}
.tab-content > .tab-pane {
  display: none;
}
.tab-content > .active {
  display: block;
}
.nav-tabs .dropdown-menu {
  margin-top: -1px;
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.navbar {
  position: relative;
  min-height: 30px;
  margin-bottom: 18px;
  border: 1px solid transparent;
}
@media (min-width: 541px) {
  .navbar {
    border-radius: 2px;
  }
}
@media (min-width: 541px) {
  .navbar-header {
    float: left;
  }
}
.navbar-collapse {
  overflow-x: visible;
  padding-right: 0px;
  padding-left: 0px;
  border-top: 1px solid transparent;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1);
  -webkit-overflow-scrolling: touch;
}
.navbar-collapse.in {
  overflow-y: auto;
}
@media (min-width: 541px) {
  .navbar-collapse {
    width: auto;
    border-top: 0;
    box-shadow: none;
  }
  .navbar-collapse.collapse {
    display: block !important;
    height: auto !important;
    padding-bottom: 0;
    overflow: visible !important;
  }
  .navbar-collapse.in {
    overflow-y: visible;
  }
  .navbar-fixed-top .navbar-collapse,
  .navbar-static-top .navbar-collapse,
  .navbar-fixed-bottom .navbar-collapse {
    padding-left: 0;
    padding-right: 0;
  }
}
.navbar-fixed-top .navbar-collapse,
.navbar-fixed-bottom .navbar-collapse {
  max-height: 340px;
}
@media (max-device-width: 540px) and (orientation: landscape) {
  .navbar-fixed-top .navbar-collapse,
  .navbar-fixed-bottom .navbar-collapse {
    max-height: 200px;
  }
}
.container > .navbar-header,
.container-fluid > .navbar-header,
.container > .navbar-collapse,
.container-fluid > .navbar-collapse {
  margin-right: 0px;
  margin-left: 0px;
}
@media (min-width: 541px) {
  .container > .navbar-header,
  .container-fluid > .navbar-header,
  .container > .navbar-collapse,
  .container-fluid > .navbar-collapse {
    margin-right: 0;
    margin-left: 0;
  }
}
.navbar-static-top {
  z-index: 1000;
  border-width: 0 0 1px;
}
@media (min-width: 541px) {
  .navbar-static-top {
    border-radius: 0;
  }
}
.navbar-fixed-top,
.navbar-fixed-bottom {
  position: fixed;
  right: 0;
  left: 0;
  z-index: 1030;
}
@media (min-width: 541px) {
  .navbar-fixed-top,
  .navbar-fixed-bottom {
    border-radius: 0;
  }
}
.navbar-fixed-top {
  top: 0;
  border-width: 0 0 1px;
}
.navbar-fixed-bottom {
  bottom: 0;
  margin-bottom: 0;
  border-width: 1px 0 0;
}
.navbar-brand {
  float: left;
  padding: 6px 0px;
  font-size: 17px;
  line-height: 18px;
  height: 30px;
}
.navbar-brand:hover,
.navbar-brand:focus {
  text-decoration: none;
}
.navbar-brand > img {
  display: block;
}
@media (min-width: 541px) {
  .navbar > .container .navbar-brand,
  .navbar > .container-fluid .navbar-brand {
    margin-left: 0px;
  }
}
.navbar-toggle {
  position: relative;
  float: right;
  margin-right: 0px;
  padding: 9px 10px;
  margin-top: -2px;
  margin-bottom: -2px;
  background-color: transparent;
  background-image: none;
  border: 1px solid transparent;
  border-radius: 2px;
}
.navbar-toggle:focus {
  outline: 0;
}
.navbar-toggle .icon-bar {
  display: block;
  width: 22px;
  height: 2px;
  border-radius: 1px;
}
.navbar-toggle .icon-bar + .icon-bar {
  margin-top: 4px;
}
@media (min-width: 541px) {
  .navbar-toggle {
    display: none;
  }
}
.navbar-nav {
  margin: 3px 0px;
}
.navbar-nav > li > a {
  padding-top: 10px;
  padding-bottom: 10px;
  line-height: 18px;
}
@media (max-width: 540px) {
  .navbar-nav .open .dropdown-menu {
    position: static;
    float: none;
    width: auto;
    margin-top: 0;
    background-color: transparent;
    border: 0;
    box-shadow: none;
  }
  .navbar-nav .open .dropdown-menu > li > a,
  .navbar-nav .open .dropdown-menu .dropdown-header {
    padding: 5px 15px 5px 25px;
  }
  .navbar-nav .open .dropdown-menu > li > a {
    line-height: 18px;
  }
  .navbar-nav .open .dropdown-menu > li > a:hover,
  .navbar-nav .open .dropdown-menu > li > a:focus {
    background-image: none;
  }
}
@media (min-width: 541px) {
  .navbar-nav {
    float: left;
    margin: 0;
  }
  .navbar-nav > li {
    float: left;
  }
  .navbar-nav > li > a {
    padding-top: 6px;
    padding-bottom: 6px;
  }
}
.navbar-form {
  margin-left: 0px;
  margin-right: 0px;
  padding: 10px 0px;
  border-top: 1px solid transparent;
  border-bottom: 1px solid transparent;
  -webkit-box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1), 0 1px 0 rgba(255, 255, 255, 0.1);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1), 0 1px 0 rgba(255, 255, 255, 0.1);
  margin-top: -1px;
  margin-bottom: -1px;
}
@media (min-width: 768px) {
  .navbar-form .form-group {
    display: inline-block;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .navbar-form .form-control {
    display: inline-block;
    width: auto;
    vertical-align: middle;
  }
  .navbar-form .form-control-static {
    display: inline-block;
  }
  .navbar-form .input-group {
    display: inline-table;
    vertical-align: middle;
  }
  .navbar-form .input-group .input-group-addon,
  .navbar-form .input-group .input-group-btn,
  .navbar-form .input-group .form-control {
    width: auto;
  }
  .navbar-form .input-group > .form-control {
    width: 100%;
  }
  .navbar-form .control-label {
    margin-bottom: 0;
    vertical-align: middle;
  }
  .navbar-form .radio,
  .navbar-form .checkbox {
    display: inline-block;
    margin-top: 0;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .navbar-form .radio label,
  .navbar-form .checkbox label {
    padding-left: 0;
  }
  .navbar-form .radio input[type="radio"],
  .navbar-form .checkbox input[type="checkbox"] {
    position: relative;
    margin-left: 0;
  }
  .navbar-form .has-feedback .form-control-feedback {
    top: 0;
  }
}
@media (max-width: 540px) {
  .navbar-form .form-group {
    margin-bottom: 5px;
  }
  .navbar-form .form-group:last-child {
    margin-bottom: 0;
  }
}
@media (min-width: 541px) {
  .navbar-form {
    width: auto;
    border: 0;
    margin-left: 0;
    margin-right: 0;
    padding-top: 0;
    padding-bottom: 0;
    -webkit-box-shadow: none;
    box-shadow: none;
  }
}
.navbar-nav > li > .dropdown-menu {
  margin-top: 0;
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.navbar-fixed-bottom .navbar-nav > li > .dropdown-menu {
  margin-bottom: 0;
  border-top-right-radius: 2px;
  border-top-left-radius: 2px;
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.navbar-btn {
  margin-top: -1px;
  margin-bottom: -1px;
}
.navbar-btn.btn-sm {
  margin-top: 0px;
  margin-bottom: 0px;
}
.navbar-btn.btn-xs {
  margin-top: 4px;
  margin-bottom: 4px;
}
.navbar-text {
  margin-top: 6px;
  margin-bottom: 6px;
}
@media (min-width: 541px) {
  .navbar-text {
    float: left;
    margin-left: 0px;
    margin-right: 0px;
  }
}
@media (min-width: 541px) {
  .navbar-left {
    float: left !important;
    float: left;
  }
  .navbar-right {
    float: right !important;
    float: right;
    margin-right: 0px;
  }
  .navbar-right ~ .navbar-right {
    margin-right: 0;
  }
}
.navbar-default {
  background-color: #f8f8f8;
  border-color: #e7e7e7;
}
.navbar-default .navbar-brand {
  color: #777;
}
.navbar-default .navbar-brand:hover,
.navbar-default .navbar-brand:focus {
  color: #5e5e5e;
  background-color: transparent;
}
.navbar-default .navbar-text {
  color: #777;
}
.navbar-default .navbar-nav > li > a {
  color: #777;
}
.navbar-default .navbar-nav > li > a:hover,
.navbar-default .navbar-nav > li > a:focus {
  color: #333;
  background-color: transparent;
}
.navbar-default .navbar-nav > .active > a,
.navbar-default .navbar-nav > .active > a:hover,
.navbar-default .navbar-nav > .active > a:focus {
  color: #555;
  background-color: #e7e7e7;
}
.navbar-default .navbar-nav > .disabled > a,
.navbar-default .navbar-nav > .disabled > a:hover,
.navbar-default .navbar-nav > .disabled > a:focus {
  color: #ccc;
  background-color: transparent;
}
.navbar-default .navbar-toggle {
  border-color: #ddd;
}
.navbar-default .navbar-toggle:hover,
.navbar-default .navbar-toggle:focus {
  background-color: #ddd;
}
.navbar-default .navbar-toggle .icon-bar {
  background-color: #888;
}
.navbar-default .navbar-collapse,
.navbar-default .navbar-form {
  border-color: #e7e7e7;
}
.navbar-default .navbar-nav > .open > a,
.navbar-default .navbar-nav > .open > a:hover,
.navbar-default .navbar-nav > .open > a:focus {
  background-color: #e7e7e7;
  color: #555;
}
@media (max-width: 540px) {
  .navbar-default .navbar-nav .open .dropdown-menu > li > a {
    color: #777;
  }
  .navbar-default .navbar-nav .open .dropdown-menu > li > a:hover,
  .navbar-default .navbar-nav .open .dropdown-menu > li > a:focus {
    color: #333;
    background-color: transparent;
  }
  .navbar-default .navbar-nav .open .dropdown-menu > .active > a,
  .navbar-default .navbar-nav .open .dropdown-menu > .active > a:hover,
  .navbar-default .navbar-nav .open .dropdown-menu > .active > a:focus {
    color: #555;
    background-color: #e7e7e7;
  }
  .navbar-default .navbar-nav .open .dropdown-menu > .disabled > a,
  .navbar-default .navbar-nav .open .dropdown-menu > .disabled > a:hover,
  .navbar-default .navbar-nav .open .dropdown-menu > .disabled > a:focus {
    color: #ccc;
    background-color: transparent;
  }
}
.navbar-default .navbar-link {
  color: #777;
}
.navbar-default .navbar-link:hover {
  color: #333;
}
.navbar-default .btn-link {
  color: #777;
}
.navbar-default .btn-link:hover,
.navbar-default .btn-link:focus {
  color: #333;
}
.navbar-default .btn-link[disabled]:hover,
fieldset[disabled] .navbar-default .btn-link:hover,
.navbar-default .btn-link[disabled]:focus,
fieldset[disabled] .navbar-default .btn-link:focus {
  color: #ccc;
}
.navbar-inverse {
  background-color: #222;
  border-color: #080808;
}
.navbar-inverse .navbar-brand {
  color: #9d9d9d;
}
.navbar-inverse .navbar-brand:hover,
.navbar-inverse .navbar-brand:focus {
  color: #fff;
  background-color: transparent;
}
.navbar-inverse .navbar-text {
  color: #9d9d9d;
}
.navbar-inverse .navbar-nav > li > a {
  color: #9d9d9d;
}
.navbar-inverse .navbar-nav > li > a:hover,
.navbar-inverse .navbar-nav > li > a:focus {
  color: #fff;
  background-color: transparent;
}
.navbar-inverse .navbar-nav > .active > a,
.navbar-inverse .navbar-nav > .active > a:hover,
.navbar-inverse .navbar-nav > .active > a:focus {
  color: #fff;
  background-color: #080808;
}
.navbar-inverse .navbar-nav > .disabled > a,
.navbar-inverse .navbar-nav > .disabled > a:hover,
.navbar-inverse .navbar-nav > .disabled > a:focus {
  color: #444;
  background-color: transparent;
}
.navbar-inverse .navbar-toggle {
  border-color: #333;
}
.navbar-inverse .navbar-toggle:hover,
.navbar-inverse .navbar-toggle:focus {
  background-color: #333;
}
.navbar-inverse .navbar-toggle .icon-bar {
  background-color: #fff;
}
.navbar-inverse .navbar-collapse,
.navbar-inverse .navbar-form {
  border-color: #101010;
}
.navbar-inverse .navbar-nav > .open > a,
.navbar-inverse .navbar-nav > .open > a:hover,
.navbar-inverse .navbar-nav > .open > a:focus {
  background-color: #080808;
  color: #fff;
}
@media (max-width: 540px) {
  .navbar-inverse .navbar-nav .open .dropdown-menu > .dropdown-header {
    border-color: #080808;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu .divider {
    background-color: #080808;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > li > a {
    color: #9d9d9d;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > li > a:hover,
  .navbar-inverse .navbar-nav .open .dropdown-menu > li > a:focus {
    color: #fff;
    background-color: transparent;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > .active > a,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .active > a:hover,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .active > a:focus {
    color: #fff;
    background-color: #080808;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > .disabled > a,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .disabled > a:hover,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .disabled > a:focus {
    color: #444;
    background-color: transparent;
  }
}
.navbar-inverse .navbar-link {
  color: #9d9d9d;
}
.navbar-inverse .navbar-link:hover {
  color: #fff;
}
.navbar-inverse .btn-link {
  color: #9d9d9d;
}
.navbar-inverse .btn-link:hover,
.navbar-inverse .btn-link:focus {
  color: #fff;
}
.navbar-inverse .btn-link[disabled]:hover,
fieldset[disabled] .navbar-inverse .btn-link:hover,
.navbar-inverse .btn-link[disabled]:focus,
fieldset[disabled] .navbar-inverse .btn-link:focus {
  color: #444;
}
.breadcrumb {
  padding: 8px 15px;
  margin-bottom: 18px;
  list-style: none;
  background-color: #f5f5f5;
  border-radius: 2px;
}
.breadcrumb > li {
  display: inline-block;
}
.breadcrumb > li + li:before {
  content: "/\00a0";
  padding: 0 5px;
  color: #5e5e5e;
}
.breadcrumb > .active {
  color: #777777;
}
.pagination {
  display: inline-block;
  padding-left: 0;
  margin: 18px 0;
  border-radius: 2px;
}
.pagination > li {
  display: inline;
}
.pagination > li > a,
.pagination > li > span {
  position: relative;
  float: left;
  padding: 6px 12px;
  line-height: 1.42857143;
  text-decoration: none;
  color: #337ab7;
  background-color: #fff;
  border: 1px solid #ddd;
  margin-left: -1px;
}
.pagination > li:first-child > a,
.pagination > li:first-child > span {
  margin-left: 0;
  border-bottom-left-radius: 2px;
  border-top-left-radius: 2px;
}
.pagination > li:last-child > a,
.pagination > li:last-child > span {
  border-bottom-right-radius: 2px;
  border-top-right-radius: 2px;
}
.pagination > li > a:hover,
.pagination > li > span:hover,
.pagination > li > a:focus,
.pagination > li > span:focus {
  z-index: 2;
  color: #23527c;
  background-color: #eeeeee;
  border-color: #ddd;
}
.pagination > .active > a,
.pagination > .active > span,
.pagination > .active > a:hover,
.pagination > .active > span:hover,
.pagination > .active > a:focus,
.pagination > .active > span:focus {
  z-index: 3;
  color: #fff;
  background-color: #337ab7;
  border-color: #337ab7;
  cursor: default;
}
.pagination > .disabled > span,
.pagination > .disabled > span:hover,
.pagination > .disabled > span:focus,
.pagination > .disabled > a,
.pagination > .disabled > a:hover,
.pagination > .disabled > a:focus {
  color: #777777;
  background-color: #fff;
  border-color: #ddd;
  cursor: not-allowed;
}
.pagination-lg > li > a,
.pagination-lg > li > span {
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
}
.pagination-lg > li:first-child > a,
.pagination-lg > li:first-child > span {
  border-bottom-left-radius: 3px;
  border-top-left-radius: 3px;
}
.pagination-lg > li:last-child > a,
.pagination-lg > li:last-child > span {
  border-bottom-right-radius: 3px;
  border-top-right-radius: 3px;
}
.pagination-sm > li > a,
.pagination-sm > li > span {
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
}
.pagination-sm > li:first-child > a,
.pagination-sm > li:first-child > span {
  border-bottom-left-radius: 1px;
  border-top-left-radius: 1px;
}
.pagination-sm > li:last-child > a,
.pagination-sm > li:last-child > span {
  border-bottom-right-radius: 1px;
  border-top-right-radius: 1px;
}
.pager {
  padding-left: 0;
  margin: 18px 0;
  list-style: none;
  text-align: center;
}
.pager li {
  display: inline;
}
.pager li > a,
.pager li > span {
  display: inline-block;
  padding: 5px 14px;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 15px;
}
.pager li > a:hover,
.pager li > a:focus {
  text-decoration: none;
  background-color: #eeeeee;
}
.pager .next > a,
.pager .next > span {
  float: right;
}
.pager .previous > a,
.pager .previous > span {
  float: left;
}
.pager .disabled > a,
.pager .disabled > a:hover,
.pager .disabled > a:focus,
.pager .disabled > span {
  color: #777777;
  background-color: #fff;
  cursor: not-allowed;
}
.label {
  display: inline;
  padding: .2em .6em .3em;
  font-size: 75%;
  font-weight: bold;
  line-height: 1;
  color: #fff;
  text-align: center;
  white-space: nowrap;
  vertical-align: baseline;
  border-radius: .25em;
}
a.label:hover,
a.label:focus {
  color: #fff;
  text-decoration: none;
  cursor: pointer;
}
.label:empty {
  display: none;
}
.btn .label {
  position: relative;
  top: -1px;
}
.label-default {
  background-color: #777777;
}
.label-default[href]:hover,
.label-default[href]:focus {
  background-color: #5e5e5e;
}
.label-primary {
  background-color: #337ab7;
}
.label-primary[href]:hover,
.label-primary[href]:focus {
  background-color: #286090;
}
.label-success {
  background-color: #5cb85c;
}
.label-success[href]:hover,
.label-success[href]:focus {
  background-color: #449d44;
}
.label-info {
  background-color: #5bc0de;
}
.label-info[href]:hover,
.label-info[href]:focus {
  background-color: #31b0d5;
}
.label-warning {
  background-color: #f0ad4e;
}
.label-warning[href]:hover,
.label-warning[href]:focus {
  background-color: #ec971f;
}
.label-danger {
  background-color: #d9534f;
}
.label-danger[href]:hover,
.label-danger[href]:focus {
  background-color: #c9302c;
}
.badge {
  display: inline-block;
  min-width: 10px;
  padding: 3px 7px;
  font-size: 12px;
  font-weight: bold;
  color: #fff;
  line-height: 1;
  vertical-align: middle;
  white-space: nowrap;
  text-align: center;
  background-color: #777777;
  border-radius: 10px;
}
.badge:empty {
  display: none;
}
.btn .badge {
  position: relative;
  top: -1px;
}
.btn-xs .badge,
.btn-group-xs > .btn .badge {
  top: 0;
  padding: 1px 5px;
}
a.badge:hover,
a.badge:focus {
  color: #fff;
  text-decoration: none;
  cursor: pointer;
}
.list-group-item.active > .badge,
.nav-pills > .active > a > .badge {
  color: #337ab7;
  background-color: #fff;
}
.list-group-item > .badge {
  float: right;
}
.list-group-item > .badge + .badge {
  margin-right: 5px;
}
.nav-pills > li > a > .badge {
  margin-left: 3px;
}
.jumbotron {
  padding-top: 30px;
  padding-bottom: 30px;
  margin-bottom: 30px;
  color: inherit;
  background-color: #eeeeee;
}
.jumbotron h1,
.jumbotron .h1 {
  color: inherit;
}
.jumbotron p {
  margin-bottom: 15px;
  font-size: 20px;
  font-weight: 200;
}
.jumbotron > hr {
  border-top-color: #d5d5d5;
}
.container .jumbotron,
.container-fluid .jumbotron {
  border-radius: 3px;
  padding-left: 0px;
  padding-right: 0px;
}
.jumbotron .container {
  max-width: 100%;
}
@media screen and (min-width: 768px) {
  .jumbotron {
    padding-top: 48px;
    padding-bottom: 48px;
  }
  .container .jumbotron,
  .container-fluid .jumbotron {
    padding-left: 60px;
    padding-right: 60px;
  }
  .jumbotron h1,
  .jumbotron .h1 {
    font-size: 59px;
  }
}
.thumbnail {
  display: block;
  padding: 4px;
  margin-bottom: 18px;
  line-height: 1.42857143;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 2px;
  -webkit-transition: border 0.2s ease-in-out;
  -o-transition: border 0.2s ease-in-out;
  transition: border 0.2s ease-in-out;
}
.thumbnail > img,
.thumbnail a > img {
  margin-left: auto;
  margin-right: auto;
}
a.thumbnail:hover,
a.thumbnail:focus,
a.thumbnail.active {
  border-color: #337ab7;
}
.thumbnail .caption {
  padding: 9px;
  color: #000;
}
.alert {
  padding: 15px;
  margin-bottom: 18px;
  border: 1px solid transparent;
  border-radius: 2px;
}
.alert h4 {
  margin-top: 0;
  color: inherit;
}
.alert .alert-link {
  font-weight: bold;
}
.alert > p,
.alert > ul {
  margin-bottom: 0;
}
.alert > p + p {
  margin-top: 5px;
}
.alert-dismissable,
.alert-dismissible {
  padding-right: 35px;
}
.alert-dismissable .close,
.alert-dismissible .close {
  position: relative;
  top: -2px;
  right: -21px;
  color: inherit;
}
.alert-success {
  background-color: #dff0d8;
  border-color: #d6e9c6;
  color: #3c763d;
}
.alert-success hr {
  border-top-color: #c9e2b3;
}
.alert-success .alert-link {
  color: #2b542c;
}
.alert-info {
  background-color: #d9edf7;
  border-color: #bce8f1;
  color: #31708f;
}
.alert-info hr {
  border-top-color: #a6e1ec;
}
.alert-info .alert-link {
  color: #245269;
}
.alert-warning {
  background-color: #fcf8e3;
  border-color: #faebcc;
  color: #8a6d3b;
}
.alert-warning hr {
  border-top-color: #f7e1b5;
}
.alert-warning .alert-link {
  color: #66512c;
}
.alert-danger {
  background-color: #f2dede;
  border-color: #ebccd1;
  color: #a94442;
}
.alert-danger hr {
  border-top-color: #e4b9c0;
}
.alert-danger .alert-link {
  color: #843534;
}
@-webkit-keyframes progress-bar-stripes {
  from {
    background-position: 40px 0;
  }
  to {
    background-position: 0 0;
  }
}
@keyframes progress-bar-stripes {
  from {
    background-position: 40px 0;
  }
  to {
    background-position: 0 0;
  }
}
.progress {
  overflow: hidden;
  height: 18px;
  margin-bottom: 18px;
  background-color: #f5f5f5;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.1);
  box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.1);
}
.progress-bar {
  float: left;
  width: 0%;
  height: 100%;
  font-size: 12px;
  line-height: 18px;
  color: #fff;
  text-align: center;
  background-color: #337ab7;
  -webkit-box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.15);
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.15);
  -webkit-transition: width 0.6s ease;
  -o-transition: width 0.6s ease;
  transition: width 0.6s ease;
}
.progress-striped .progress-bar,
.progress-bar-striped {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-size: 40px 40px;
}
.progress.active .progress-bar,
.progress-bar.active {
  -webkit-animation: progress-bar-stripes 2s linear infinite;
  -o-animation: progress-bar-stripes 2s linear infinite;
  animation: progress-bar-stripes 2s linear infinite;
}
.progress-bar-success {
  background-color: #5cb85c;
}
.progress-striped .progress-bar-success {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.progress-bar-info {
  background-color: #5bc0de;
}
.progress-striped .progress-bar-info {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.progress-bar-warning {
  background-color: #f0ad4e;
}
.progress-striped .progress-bar-warning {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.progress-bar-danger {
  background-color: #d9534f;
}
.progress-striped .progress-bar-danger {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.media {
  margin-top: 15px;
}
.media:first-child {
  margin-top: 0;
}
.media,
.media-body {
  zoom: 1;
  overflow: hidden;
}
.media-body {
  width: 10000px;
}
.media-object {
  display: block;
}
.media-object.img-thumbnail {
  max-width: none;
}
.media-right,
.media > .pull-right {
  padding-left: 10px;
}
.media-left,
.media > .pull-left {
  padding-right: 10px;
}
.media-left,
.media-right,
.media-body {
  display: table-cell;
  vertical-align: top;
}
.media-middle {
  vertical-align: middle;
}
.media-bottom {
  vertical-align: bottom;
}
.media-heading {
  margin-top: 0;
  margin-bottom: 5px;
}
.media-list {
  padding-left: 0;
  list-style: none;
}
.list-group {
  margin-bottom: 20px;
  padding-left: 0;
}
.list-group-item {
  position: relative;
  display: block;
  padding: 10px 15px;
  margin-bottom: -1px;
  background-color: #fff;
  border: 1px solid #ddd;
}
.list-group-item:first-child {
  border-top-right-radius: 2px;
  border-top-left-radius: 2px;
}
.list-group-item:last-child {
  margin-bottom: 0;
  border-bottom-right-radius: 2px;
  border-bottom-left-radius: 2px;
}
a.list-group-item,
button.list-group-item {
  color: #555;
}
a.list-group-item .list-group-item-heading,
button.list-group-item .list-group-item-heading {
  color: #333;
}
a.list-group-item:hover,
button.list-group-item:hover,
a.list-group-item:focus,
button.list-group-item:focus {
  text-decoration: none;
  color: #555;
  background-color: #f5f5f5;
}
button.list-group-item {
  width: 100%;
  text-align: left;
}
.list-group-item.disabled,
.list-group-item.disabled:hover,
.list-group-item.disabled:focus {
  background-color: #eeeeee;
  color: #777777;
  cursor: not-allowed;
}
.list-group-item.disabled .list-group-item-heading,
.list-group-item.disabled:hover .list-group-item-heading,
.list-group-item.disabled:focus .list-group-item-heading {
  color: inherit;
}
.list-group-item.disabled .list-group-item-text,
.list-group-item.disabled:hover .list-group-item-text,
.list-group-item.disabled:focus .list-group-item-text {
  color: #777777;
}
.list-group-item.active,
.list-group-item.active:hover,
.list-group-item.active:focus {
  z-index: 2;
  color: #fff;
  background-color: #337ab7;
  border-color: #337ab7;
}
.list-group-item.active .list-group-item-heading,
.list-group-item.active:hover .list-group-item-heading,
.list-group-item.active:focus .list-group-item-heading,
.list-group-item.active .list-group-item-heading > small,
.list-group-item.active:hover .list-group-item-heading > small,
.list-group-item.active:focus .list-group-item-heading > small,
.list-group-item.active .list-group-item-heading > .small,
.list-group-item.active:hover .list-group-item-heading > .small,
.list-group-item.active:focus .list-group-item-heading > .small {
  color: inherit;
}
.list-group-item.active .list-group-item-text,
.list-group-item.active:hover .list-group-item-text,
.list-group-item.active:focus .list-group-item-text {
  color: #c7ddef;
}
.list-group-item-success {
  color: #3c763d;
  background-color: #dff0d8;
}
a.list-group-item-success,
button.list-group-item-success {
  color: #3c763d;
}
a.list-group-item-success .list-group-item-heading,
button.list-group-item-success .list-group-item-heading {
  color: inherit;
}
a.list-group-item-success:hover,
button.list-group-item-success:hover,
a.list-group-item-success:focus,
button.list-group-item-success:focus {
  color: #3c763d;
  background-color: #d0e9c6;
}
a.list-group-item-success.active,
button.list-group-item-success.active,
a.list-group-item-success.active:hover,
button.list-group-item-success.active:hover,
a.list-group-item-success.active:focus,
button.list-group-item-success.active:focus {
  color: #fff;
  background-color: #3c763d;
  border-color: #3c763d;
}
.list-group-item-info {
  color: #31708f;
  background-color: #d9edf7;
}
a.list-group-item-info,
button.list-group-item-info {
  color: #31708f;
}
a.list-group-item-info .list-group-item-heading,
button.list-group-item-info .list-group-item-heading {
  color: inherit;
}
a.list-group-item-info:hover,
button.list-group-item-info:hover,
a.list-group-item-info:focus,
button.list-group-item-info:focus {
  color: #31708f;
  background-color: #c4e3f3;
}
a.list-group-item-info.active,
button.list-group-item-info.active,
a.list-group-item-info.active:hover,
button.list-group-item-info.active:hover,
a.list-group-item-info.active:focus,
button.list-group-item-info.active:focus {
  color: #fff;
  background-color: #31708f;
  border-color: #31708f;
}
.list-group-item-warning {
  color: #8a6d3b;
  background-color: #fcf8e3;
}
a.list-group-item-warning,
button.list-group-item-warning {
  color: #8a6d3b;
}
a.list-group-item-warning .list-group-item-heading,
button.list-group-item-warning .list-group-item-heading {
  color: inherit;
}
a.list-group-item-warning:hover,
button.list-group-item-warning:hover,
a.list-group-item-warning:focus,
button.list-group-item-warning:focus {
  color: #8a6d3b;
  background-color: #faf2cc;
}
a.list-group-item-warning.active,
button.list-group-item-warning.active,
a.list-group-item-warning.active:hover,
button.list-group-item-warning.active:hover,
a.list-group-item-warning.active:focus,
button.list-group-item-warning.active:focus {
  color: #fff;
  background-color: #8a6d3b;
  border-color: #8a6d3b;
}
.list-group-item-danger {
  color: #a94442;
  background-color: #f2dede;
}
a.list-group-item-danger,
button.list-group-item-danger {
  color: #a94442;
}
a.list-group-item-danger .list-group-item-heading,
button.list-group-item-danger .list-group-item-heading {
  color: inherit;
}
a.list-group-item-danger:hover,
button.list-group-item-danger:hover,
a.list-group-item-danger:focus,
button.list-group-item-danger:focus {
  color: #a94442;
  background-color: #ebcccc;
}
a.list-group-item-danger.active,
button.list-group-item-danger.active,
a.list-group-item-danger.active:hover,
button.list-group-item-danger.active:hover,
a.list-group-item-danger.active:focus,
button.list-group-item-danger.active:focus {
  color: #fff;
  background-color: #a94442;
  border-color: #a94442;
}
.list-group-item-heading {
  margin-top: 0;
  margin-bottom: 5px;
}
.list-group-item-text {
  margin-bottom: 0;
  line-height: 1.3;
}
.panel {
  margin-bottom: 18px;
  background-color: #fff;
  border: 1px solid transparent;
  border-radius: 2px;
  -webkit-box-shadow: 0 1px 1px rgba(0, 0, 0, 0.05);
  box-shadow: 0 1px 1px rgba(0, 0, 0, 0.05);
}
.panel-body {
  padding: 15px;
}
.panel-heading {
  padding: 10px 15px;
  border-bottom: 1px solid transparent;
  border-top-right-radius: 1px;
  border-top-left-radius: 1px;
}
.panel-heading > .dropdown .dropdown-toggle {
  color: inherit;
}
.panel-title {
  margin-top: 0;
  margin-bottom: 0;
  font-size: 15px;
  color: inherit;
}
.panel-title > a,
.panel-title > small,
.panel-title > .small,
.panel-title > small > a,
.panel-title > .small > a {
  color: inherit;
}
.panel-footer {
  padding: 10px 15px;
  background-color: #f5f5f5;
  border-top: 1px solid #ddd;
  border-bottom-right-radius: 1px;
  border-bottom-left-radius: 1px;
}
.panel > .list-group,
.panel > .panel-collapse > .list-group {
  margin-bottom: 0;
}
.panel > .list-group .list-group-item,
.panel > .panel-collapse > .list-group .list-group-item {
  border-width: 1px 0;
  border-radius: 0;
}
.panel > .list-group:first-child .list-group-item:first-child,
.panel > .panel-collapse > .list-group:first-child .list-group-item:first-child {
  border-top: 0;
  border-top-right-radius: 1px;
  border-top-left-radius: 1px;
}
.panel > .list-group:last-child .list-group-item:last-child,
.panel > .panel-collapse > .list-group:last-child .list-group-item:last-child {
  border-bottom: 0;
  border-bottom-right-radius: 1px;
  border-bottom-left-radius: 1px;
}
.panel > .panel-heading + .panel-collapse > .list-group .list-group-item:first-child {
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.panel-heading + .list-group .list-group-item:first-child {
  border-top-width: 0;
}
.list-group + .panel-footer {
  border-top-width: 0;
}
.panel > .table,
.panel > .table-responsive > .table,
.panel > .panel-collapse > .table {
  margin-bottom: 0;
}
.panel > .table caption,
.panel > .table-responsive > .table caption,
.panel > .panel-collapse > .table caption {
  padding-left: 15px;
  padding-right: 15px;
}
.panel > .table:first-child,
.panel > .table-responsive:first-child > .table:first-child {
  border-top-right-radius: 1px;
  border-top-left-radius: 1px;
}
.panel > .table:first-child > thead:first-child > tr:first-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child,
.panel > .table:first-child > tbody:first-child > tr:first-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child {
  border-top-left-radius: 1px;
  border-top-right-radius: 1px;
}
.panel > .table:first-child > thead:first-child > tr:first-child td:first-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child td:first-child,
.panel > .table:first-child > tbody:first-child > tr:first-child td:first-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child td:first-child,
.panel > .table:first-child > thead:first-child > tr:first-child th:first-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child th:first-child,
.panel > .table:first-child > tbody:first-child > tr:first-child th:first-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child th:first-child {
  border-top-left-radius: 1px;
}
.panel > .table:first-child > thead:first-child > tr:first-child td:last-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child td:last-child,
.panel > .table:first-child > tbody:first-child > tr:first-child td:last-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child td:last-child,
.panel > .table:first-child > thead:first-child > tr:first-child th:last-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child th:last-child,
.panel > .table:first-child > tbody:first-child > tr:first-child th:last-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child th:last-child {
  border-top-right-radius: 1px;
}
.panel > .table:last-child,
.panel > .table-responsive:last-child > .table:last-child {
  border-bottom-right-radius: 1px;
  border-bottom-left-radius: 1px;
}
.panel > .table:last-child > tbody:last-child > tr:last-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child {
  border-bottom-left-radius: 1px;
  border-bottom-right-radius: 1px;
}
.panel > .table:last-child > tbody:last-child > tr:last-child td:first-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child td:first-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child td:first-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child td:first-child,
.panel > .table:last-child > tbody:last-child > tr:last-child th:first-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child th:first-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child th:first-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child th:first-child {
  border-bottom-left-radius: 1px;
}
.panel > .table:last-child > tbody:last-child > tr:last-child td:last-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child td:last-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child td:last-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child td:last-child,
.panel > .table:last-child > tbody:last-child > tr:last-child th:last-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child th:last-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child th:last-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child th:last-child {
  border-bottom-right-radius: 1px;
}
.panel > .panel-body + .table,
.panel > .panel-body + .table-responsive,
.panel > .table + .panel-body,
.panel > .table-responsive + .panel-body {
  border-top: 1px solid #ddd;
}
.panel > .table > tbody:first-child > tr:first-child th,
.panel > .table > tbody:first-child > tr:first-child td {
  border-top: 0;
}
.panel > .table-bordered,
.panel > .table-responsive > .table-bordered {
  border: 0;
}
.panel > .table-bordered > thead > tr > th:first-child,
.panel > .table-responsive > .table-bordered > thead > tr > th:first-child,
.panel > .table-bordered > tbody > tr > th:first-child,
.panel > .table-responsive > .table-bordered > tbody > tr > th:first-child,
.panel > .table-bordered > tfoot > tr > th:first-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > th:first-child,
.panel > .table-bordered > thead > tr > td:first-child,
.panel > .table-responsive > .table-bordered > thead > tr > td:first-child,
.panel > .table-bordered > tbody > tr > td:first-child,
.panel > .table-responsive > .table-bordered > tbody > tr > td:first-child,
.panel > .table-bordered > tfoot > tr > td:first-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > td:first-child {
  border-left: 0;
}
.panel > .table-bordered > thead > tr > th:last-child,
.panel > .table-responsive > .table-bordered > thead > tr > th:last-child,
.panel > .table-bordered > tbody > tr > th:last-child,
.panel > .table-responsive > .table-bordered > tbody > tr > th:last-child,
.panel > .table-bordered > tfoot > tr > th:last-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > th:last-child,
.panel > .table-bordered > thead > tr > td:last-child,
.panel > .table-responsive > .table-bordered > thead > tr > td:last-child,
.panel > .table-bordered > tbody > tr > td:last-child,
.panel > .table-responsive > .table-bordered > tbody > tr > td:last-child,
.panel > .table-bordered > tfoot > tr > td:last-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > td:last-child {
  border-right: 0;
}
.panel > .table-bordered > thead > tr:first-child > td,
.panel > .table-responsive > .table-bordered > thead > tr:first-child > td,
.panel > .table-bordered > tbody > tr:first-child > td,
.panel > .table-responsive > .table-bordered > tbody > tr:first-child > td,
.panel > .table-bordered > thead > tr:first-child > th,
.panel > .table-responsive > .table-bordered > thead > tr:first-child > th,
.panel > .table-bordered > tbody > tr:first-child > th,
.panel > .table-responsive > .table-bordered > tbody > tr:first-child > th {
  border-bottom: 0;
}
.panel > .table-bordered > tbody > tr:last-child > td,
.panel > .table-responsive > .table-bordered > tbody > tr:last-child > td,
.panel > .table-bordered > tfoot > tr:last-child > td,
.panel > .table-responsive > .table-bordered > tfoot > tr:last-child > td,
.panel > .table-bordered > tbody > tr:last-child > th,
.panel > .table-responsive > .table-bordered > tbody > tr:last-child > th,
.panel > .table-bordered > tfoot > tr:last-child > th,
.panel > .table-responsive > .table-bordered > tfoot > tr:last-child > th {
  border-bottom: 0;
}
.panel > .table-responsive {
  border: 0;
  margin-bottom: 0;
}
.panel-group {
  margin-bottom: 18px;
}
.panel-group .panel {
  margin-bottom: 0;
  border-radius: 2px;
}
.panel-group .panel + .panel {
  margin-top: 5px;
}
.panel-group .panel-heading {
  border-bottom: 0;
}
.panel-group .panel-heading + .panel-collapse > .panel-body,
.panel-group .panel-heading + .panel-collapse > .list-group {
  border-top: 1px solid #ddd;
}
.panel-group .panel-footer {
  border-top: 0;
}
.panel-group .panel-footer + .panel-collapse .panel-body {
  border-bottom: 1px solid #ddd;
}
.panel-default {
  border-color: #ddd;
}
.panel-default > .panel-heading {
  color: #333333;
  background-color: #f5f5f5;
  border-color: #ddd;
}
.panel-default > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #ddd;
}
.panel-default > .panel-heading .badge {
  color: #f5f5f5;
  background-color: #333333;
}
.panel-default > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #ddd;
}
.panel-primary {
  border-color: #337ab7;
}
.panel-primary > .panel-heading {
  color: #fff;
  background-color: #337ab7;
  border-color: #337ab7;
}
.panel-primary > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #337ab7;
}
.panel-primary > .panel-heading .badge {
  color: #337ab7;
  background-color: #fff;
}
.panel-primary > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #337ab7;
}
.panel-success {
  border-color: #d6e9c6;
}
.panel-success > .panel-heading {
  color: #3c763d;
  background-color: #dff0d8;
  border-color: #d6e9c6;
}
.panel-success > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #d6e9c6;
}
.panel-success > .panel-heading .badge {
  color: #dff0d8;
  background-color: #3c763d;
}
.panel-success > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #d6e9c6;
}
.panel-info {
  border-color: #bce8f1;
}
.panel-info > .panel-heading {
  color: #31708f;
  background-color: #d9edf7;
  border-color: #bce8f1;
}
.panel-info > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #bce8f1;
}
.panel-info > .panel-heading .badge {
  color: #d9edf7;
  background-color: #31708f;
}
.panel-info > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #bce8f1;
}
.panel-warning {
  border-color: #faebcc;
}
.panel-warning > .panel-heading {
  color: #8a6d3b;
  background-color: #fcf8e3;
  border-color: #faebcc;
}
.panel-warning > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #faebcc;
}
.panel-warning > .panel-heading .badge {
  color: #fcf8e3;
  background-color: #8a6d3b;
}
.panel-warning > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #faebcc;
}
.panel-danger {
  border-color: #ebccd1;
}
.panel-danger > .panel-heading {
  color: #a94442;
  background-color: #f2dede;
  border-color: #ebccd1;
}
.panel-danger > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #ebccd1;
}
.panel-danger > .panel-heading .badge {
  color: #f2dede;
  background-color: #a94442;
}
.panel-danger > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #ebccd1;
}
.embed-responsive {
  position: relative;
  display: block;
  height: 0;
  padding: 0;
  overflow: hidden;
}
.embed-responsive .embed-responsive-item,
.embed-responsive iframe,
.embed-responsive embed,
.embed-responsive object,
.embed-responsive video {
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  height: 100%;
  width: 100%;
  border: 0;
}
.embed-responsive-16by9 {
  padding-bottom: 56.25%;
}
.embed-responsive-4by3 {
  padding-bottom: 75%;
}
.well {
  min-height: 20px;
  padding: 19px;
  margin-bottom: 20px;
  background-color: #f5f5f5;
  border: 1px solid #e3e3e3;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.05);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.05);
}
.well blockquote {
  border-color: #ddd;
  border-color: rgba(0, 0, 0, 0.15);
}
.well-lg {
  padding: 24px;
  border-radius: 3px;
}
.well-sm {
  padding: 9px;
  border-radius: 1px;
}
.close {
  float: right;
  font-size: 19.5px;
  font-weight: bold;
  line-height: 1;
  color: #000;
  text-shadow: 0 1px 0 #fff;
  opacity: 0.2;
  filter: alpha(opacity=20);
}
.close:hover,
.close:focus {
  color: #000;
  text-decoration: none;
  cursor: pointer;
  opacity: 0.5;
  filter: alpha(opacity=50);
}
button.close {
  padding: 0;
  cursor: pointer;
  background: transparent;
  border: 0;
  -webkit-appearance: none;
}
.modal-open {
  overflow: hidden;
}
.modal {
  display: none;
  overflow: hidden;
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  z-index: 1050;
  -webkit-overflow-scrolling: touch;
  outline: 0;
}
.modal.fade .modal-dialog {
  -webkit-transform: translate(0, -25%);
  -ms-transform: translate(0, -25%);
  -o-transform: translate(0, -25%);
  transform: translate(0, -25%);
  -webkit-transition: -webkit-transform 0.3s ease-out;
  -moz-transition: -moz-transform 0.3s ease-out;
  -o-transition: -o-transform 0.3s ease-out;
  transition: transform 0.3s ease-out;
}
.modal.in .modal-dialog {
  -webkit-transform: translate(0, 0);
  -ms-transform: translate(0, 0);
  -o-transform: translate(0, 0);
  transform: translate(0, 0);
}
.modal-open .modal {
  overflow-x: hidden;
  overflow-y: auto;
}
.modal-dialog {
  position: relative;
  width: auto;
  margin: 10px;
}
.modal-content {
  position: relative;
  background-color: #fff;
  border: 1px solid #999;
  border: 1px solid rgba(0, 0, 0, 0.2);
  border-radius: 3px;
  -webkit-box-shadow: 0 3px 9px rgba(0, 0, 0, 0.5);
  box-shadow: 0 3px 9px rgba(0, 0, 0, 0.5);
  background-clip: padding-box;
  outline: 0;
}
.modal-backdrop {
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  z-index: 1040;
  background-color: #000;
}
.modal-backdrop.fade {
  opacity: 0;
  filter: alpha(opacity=0);
}
.modal-backdrop.in {
  opacity: 0.5;
  filter: alpha(opacity=50);
}
.modal-header {
  padding: 15px;
  border-bottom: 1px solid #e5e5e5;
}
.modal-header .close {
  margin-top: -2px;
}
.modal-title {
  margin: 0;
  line-height: 1.42857143;
}
.modal-body {
  position: relative;
  padding: 15px;
}
.modal-footer {
  padding: 15px;
  text-align: right;
  border-top: 1px solid #e5e5e5;
}
.modal-footer .btn + .btn {
  margin-left: 5px;
  margin-bottom: 0;
}
.modal-footer .btn-group .btn + .btn {
  margin-left: -1px;
}
.modal-footer .btn-block + .btn-block {
  margin-left: 0;
}
.modal-scrollbar-measure {
  position: absolute;
  top: -9999px;
  width: 50px;
  height: 50px;
  overflow: scroll;
}
@media (min-width: 768px) {
  .modal-dialog {
    width: 600px;
    margin: 30px auto;
  }
  .modal-content {
    -webkit-box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
  }
  .modal-sm {
    width: 300px;
  }
}
@media (min-width: 992px) {
  .modal-lg {
    width: 900px;
  }
}
.tooltip {
  position: absolute;
  z-index: 1070;
  display: block;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-style: normal;
  font-weight: normal;
  letter-spacing: normal;
  line-break: auto;
  line-height: 1.42857143;
  text-align: left;
  text-align: start;
  text-decoration: none;
  text-shadow: none;
  text-transform: none;
  white-space: normal;
  word-break: normal;
  word-spacing: normal;
  word-wrap: normal;
  font-size: 12px;
  opacity: 0;
  filter: alpha(opacity=0);
}
.tooltip.in {
  opacity: 0.9;
  filter: alpha(opacity=90);
}
.tooltip.top {
  margin-top: -3px;
  padding: 5px 0;
}
.tooltip.right {
  margin-left: 3px;
  padding: 0 5px;
}
.tooltip.bottom {
  margin-top: 3px;
  padding: 5px 0;
}
.tooltip.left {
  margin-left: -3px;
  padding: 0 5px;
}
.tooltip-inner {
  max-width: 200px;
  padding: 3px 8px;
  color: #fff;
  text-align: center;
  background-color: #000;
  border-radius: 2px;
}
.tooltip-arrow {
  position: absolute;
  width: 0;
  height: 0;
  border-color: transparent;
  border-style: solid;
}
.tooltip.top .tooltip-arrow {
  bottom: 0;
  left: 50%;
  margin-left: -5px;
  border-width: 5px 5px 0;
  border-top-color: #000;
}
.tooltip.top-left .tooltip-arrow {
  bottom: 0;
  right: 5px;
  margin-bottom: -5px;
  border-width: 5px 5px 0;
  border-top-color: #000;
}
.tooltip.top-right .tooltip-arrow {
  bottom: 0;
  left: 5px;
  margin-bottom: -5px;
  border-width: 5px 5px 0;
  border-top-color: #000;
}
.tooltip.right .tooltip-arrow {
  top: 50%;
  left: 0;
  margin-top: -5px;
  border-width: 5px 5px 5px 0;
  border-right-color: #000;
}
.tooltip.left .tooltip-arrow {
  top: 50%;
  right: 0;
  margin-top: -5px;
  border-width: 5px 0 5px 5px;
  border-left-color: #000;
}
.tooltip.bottom .tooltip-arrow {
  top: 0;
  left: 50%;
  margin-left: -5px;
  border-width: 0 5px 5px;
  border-bottom-color: #000;
}
.tooltip.bottom-left .tooltip-arrow {
  top: 0;
  right: 5px;
  margin-top: -5px;
  border-width: 0 5px 5px;
  border-bottom-color: #000;
}
.tooltip.bottom-right .tooltip-arrow {
  top: 0;
  left: 5px;
  margin-top: -5px;
  border-width: 0 5px 5px;
  border-bottom-color: #000;
}
.popover {
  position: absolute;
  top: 0;
  left: 0;
  z-index: 1060;
  display: none;
  max-width: 276px;
  padding: 1px;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-style: normal;
  font-weight: normal;
  letter-spacing: normal;
  line-break: auto;
  line-height: 1.42857143;
  text-align: left;
  text-align: start;
  text-decoration: none;
  text-shadow: none;
  text-transform: none;
  white-space: normal;
  word-break: normal;
  word-spacing: normal;
  word-wrap: normal;
  font-size: 13px;
  background-color: #fff;
  background-clip: padding-box;
  border: 1px solid #ccc;
  border: 1px solid rgba(0, 0, 0, 0.2);
  border-radius: 3px;
  -webkit-box-shadow: 0 5px 10px rgba(0, 0, 0, 0.2);
  box-shadow: 0 5px 10px rgba(0, 0, 0, 0.2);
}
.popover.top {
  margin-top: -10px;
}
.popover.right {
  margin-left: 10px;
}
.popover.bottom {
  margin-top: 10px;
}
.popover.left {
  margin-left: -10px;
}
.popover-title {
  margin: 0;
  padding: 8px 14px;
  font-size: 13px;
  background-color: #f7f7f7;
  border-bottom: 1px solid #ebebeb;
  border-radius: 2px 2px 0 0;
}
.popover-content {
  padding: 9px 14px;
}
.popover > .arrow,
.popover > .arrow:after {
  position: absolute;
  display: block;
  width: 0;
  height: 0;
  border-color: transparent;
  border-style: solid;
}
.popover > .arrow {
  border-width: 11px;
}
.popover > .arrow:after {
  border-width: 10px;
  content: "";
}
.popover.top > .arrow {
  left: 50%;
  margin-left: -11px;
  border-bottom-width: 0;
  border-top-color: #999999;
  border-top-color: rgba(0, 0, 0, 0.25);
  bottom: -11px;
}
.popover.top > .arrow:after {
  content: " ";
  bottom: 1px;
  margin-left: -10px;
  border-bottom-width: 0;
  border-top-color: #fff;
}
.popover.right > .arrow {
  top: 50%;
  left: -11px;
  margin-top: -11px;
  border-left-width: 0;
  border-right-color: #999999;
  border-right-color: rgba(0, 0, 0, 0.25);
}
.popover.right > .arrow:after {
  content: " ";
  left: 1px;
  bottom: -10px;
  border-left-width: 0;
  border-right-color: #fff;
}
.popover.bottom > .arrow {
  left: 50%;
  margin-left: -11px;
  border-top-width: 0;
  border-bottom-color: #999999;
  border-bottom-color: rgba(0, 0, 0, 0.25);
  top: -11px;
}
.popover.bottom > .arrow:after {
  content: " ";
  top: 1px;
  margin-left: -10px;
  border-top-width: 0;
  border-bottom-color: #fff;
}
.popover.left > .arrow {
  top: 50%;
  right: -11px;
  margin-top: -11px;
  border-right-width: 0;
  border-left-color: #999999;
  border-left-color: rgba(0, 0, 0, 0.25);
}
.popover.left > .arrow:after {
  content: " ";
  right: 1px;
  border-right-width: 0;
  border-left-color: #fff;
  bottom: -10px;
}
.carousel {
  position: relative;
}
.carousel-inner {
  position: relative;
  overflow: hidden;
  width: 100%;
}
.carousel-inner > .item {
  display: none;
  position: relative;
  -webkit-transition: 0.6s ease-in-out left;
  -o-transition: 0.6s ease-in-out left;
  transition: 0.6s ease-in-out left;
}
.carousel-inner > .item > img,
.carousel-inner > .item > a > img {
  line-height: 1;
}
@media all and (transform-3d), (-webkit-transform-3d) {
  .carousel-inner > .item {
    -webkit-transition: -webkit-transform 0.6s ease-in-out;
    -moz-transition: -moz-transform 0.6s ease-in-out;
    -o-transition: -o-transform 0.6s ease-in-out;
    transition: transform 0.6s ease-in-out;
    -webkit-backface-visibility: hidden;
    -moz-backface-visibility: hidden;
    backface-visibility: hidden;
    -webkit-perspective: 1000px;
    -moz-perspective: 1000px;
    perspective: 1000px;
  }
  .carousel-inner > .item.next,
  .carousel-inner > .item.active.right {
    -webkit-transform: translate3d(100%, 0, 0);
    transform: translate3d(100%, 0, 0);
    left: 0;
  }
  .carousel-inner > .item.prev,
  .carousel-inner > .item.active.left {
    -webkit-transform: translate3d(-100%, 0, 0);
    transform: translate3d(-100%, 0, 0);
    left: 0;
  }
  .carousel-inner > .item.next.left,
  .carousel-inner > .item.prev.right,
  .carousel-inner > .item.active {
    -webkit-transform: translate3d(0, 0, 0);
    transform: translate3d(0, 0, 0);
    left: 0;
  }
}
.carousel-inner > .active,
.carousel-inner > .next,
.carousel-inner > .prev {
  display: block;
}
.carousel-inner > .active {
  left: 0;
}
.carousel-inner > .next,
.carousel-inner > .prev {
  position: absolute;
  top: 0;
  width: 100%;
}
.carousel-inner > .next {
  left: 100%;
}
.carousel-inner > .prev {
  left: -100%;
}
.carousel-inner > .next.left,
.carousel-inner > .prev.right {
  left: 0;
}
.carousel-inner > .active.left {
  left: -100%;
}
.carousel-inner > .active.right {
  left: 100%;
}
.carousel-control {
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  width: 15%;
  opacity: 0.5;
  filter: alpha(opacity=50);
  font-size: 20px;
  color: #fff;
  text-align: center;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
  background-color: rgba(0, 0, 0, 0);
}
.carousel-control.left {
  background-image: -webkit-linear-gradient(left, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.0001) 100%);
  background-image: -o-linear-gradient(left, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.0001) 100%);
  background-image: linear-gradient(to right, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.0001) 100%);
  background-repeat: repeat-x;
  filter: progid:DXImageTransform.Microsoft.gradient(startColorstr='#80000000', endColorstr='#00000000', GradientType=1);
}
.carousel-control.right {
  left: auto;
  right: 0;
  background-image: -webkit-linear-gradient(left, rgba(0, 0, 0, 0.0001) 0%, rgba(0, 0, 0, 0.5) 100%);
  background-image: -o-linear-gradient(left, rgba(0, 0, 0, 0.0001) 0%, rgba(0, 0, 0, 0.5) 100%);
  background-image: linear-gradient(to right, rgba(0, 0, 0, 0.0001) 0%, rgba(0, 0, 0, 0.5) 100%);
  background-repeat: repeat-x;
  filter: progid:DXImageTransform.Microsoft.gradient(startColorstr='#00000000', endColorstr='#80000000', GradientType=1);
}
.carousel-control:hover,
.carousel-control:focus {
  outline: 0;
  color: #fff;
  text-decoration: none;
  opacity: 0.9;
  filter: alpha(opacity=90);
}
.carousel-control .icon-prev,
.carousel-control .icon-next,
.carousel-control .glyphicon-chevron-left,
.carousel-control .glyphicon-chevron-right {
  position: absolute;
  top: 50%;
  margin-top: -10px;
  z-index: 5;
  display: inline-block;
}
.carousel-control .icon-prev,
.carousel-control .glyphicon-chevron-left {
  left: 50%;
  margin-left: -10px;
}
.carousel-control .icon-next,
.carousel-control .glyphicon-chevron-right {
  right: 50%;
  margin-right: -10px;
}
.carousel-control .icon-prev,
.carousel-control .icon-next {
  width: 20px;
  height: 20px;
  line-height: 1;
  font-family: serif;
}
.carousel-control .icon-prev:before {
  content: '\2039';
}
.carousel-control .icon-next:before {
  content: '\203a';
}
.carousel-indicators {
  position: absolute;
  bottom: 10px;
  left: 50%;
  z-index: 15;
  width: 60%;
  margin-left: -30%;
  padding-left: 0;
  list-style: none;
  text-align: center;
}
.carousel-indicators li {
  display: inline-block;
  width: 10px;
  height: 10px;
  margin: 1px;
  text-indent: -999px;
  border: 1px solid #fff;
  border-radius: 10px;
  cursor: pointer;
  background-color: #000 \9;
  background-color: rgba(0, 0, 0, 0);
}
.carousel-indicators .active {
  margin: 0;
  width: 12px;
  height: 12px;
  background-color: #fff;
}
.carousel-caption {
  position: absolute;
  left: 15%;
  right: 15%;
  bottom: 20px;
  z-index: 10;
  padding-top: 20px;
  padding-bottom: 20px;
  color: #fff;
  text-align: center;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
}
.carousel-caption .btn {
  text-shadow: none;
}
@media screen and (min-width: 768px) {
  .carousel-control .glyphicon-chevron-left,
  .carousel-control .glyphicon-chevron-right,
  .carousel-control .icon-prev,
  .carousel-control .icon-next {
    width: 30px;
    height: 30px;
    margin-top: -10px;
    font-size: 30px;
  }
  .carousel-control .glyphicon-chevron-left,
  .carousel-control .icon-prev {
    margin-left: -10px;
  }
  .carousel-control .glyphicon-chevron-right,
  .carousel-control .icon-next {
    margin-right: -10px;
  }
  .carousel-caption {
    left: 20%;
    right: 20%;
    padding-bottom: 30px;
  }
  .carousel-indicators {
    bottom: 20px;
  }
}
.clearfix:before,
.clearfix:after,
.dl-horizontal dd:before,
.dl-horizontal dd:after,
.container:before,
.container:after,
.container-fluid:before,
.container-fluid:after,
.row:before,
.row:after,
.form-horizontal .form-group:before,
.form-horizontal .form-group:after,
.btn-toolbar:before,
.btn-toolbar:after,
.btn-group-vertical > .btn-group:before,
.btn-group-vertical > .btn-group:after,
.nav:before,
.nav:after,
.navbar:before,
.navbar:after,
.navbar-header:before,
.navbar-header:after,
.navbar-collapse:before,
.navbar-collapse:after,
.pager:before,
.pager:after,
.panel-body:before,
.panel-body:after,
.modal-header:before,
.modal-header:after,
.modal-footer:before,
.modal-footer:after,
.item_buttons:before,
.item_buttons:after {
  content: " ";
  display: table;
}
.clearfix:after,
.dl-horizontal dd:after,
.container:after,
.container-fluid:after,
.row:after,
.form-horizontal .form-group:after,
.btn-toolbar:after,
.btn-group-vertical > .btn-group:after,
.nav:after,
.navbar:after,
.navbar-header:after,
.navbar-collapse:after,
.pager:after,
.panel-body:after,
.modal-header:after,
.modal-footer:after,
.item_buttons:after {
  clear: both;
}
.center-block {
  display: block;
  margin-left: auto;
  margin-right: auto;
}
.pull-right {
  float: right !important;
}
.pull-left {
  float: left !important;
}
.hide {
  display: none !important;
}
.show {
  display: block !important;
}
.invisible {
  visibility: hidden;
}
.text-hide {
  font: 0/0 a;
  color: transparent;
  text-shadow: none;
  background-color: transparent;
  border: 0;
}
.hidden {
  display: none !important;
}
.affix {
  position: fixed;
}
@-ms-viewport {
  width: device-width;
}
.visible-xs,
.visible-sm,
.visible-md,
.visible-lg {
  display: none !important;
}
.visible-xs-block,
.visible-xs-inline,
.visible-xs-inline-block,
.visible-sm-block,
.visible-sm-inline,
.visible-sm-inline-block,
.visible-md-block,
.visible-md-inline,
.visible-md-inline-block,
.visible-lg-block,
.visible-lg-inline,
.visible-lg-inline-block {
  display: none !important;
}
@media (max-width: 767px) {
  .visible-xs {
    display: block !important;
  }
  table.visible-xs {
    display: table !important;
  }
  tr.visible-xs {
    display: table-row !important;
  }
  th.visible-xs,
  td.visible-xs {
    display: table-cell !important;
  }
}
@media (max-width: 767px) {
  .visible-xs-block {
    display: block !important;
  }
}
@media (max-width: 767px) {
  .visible-xs-inline {
    display: inline !important;
  }
}
@media (max-width: 767px) {
  .visible-xs-inline-block {
    display: inline-block !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm {
    display: block !important;
  }
  table.visible-sm {
    display: table !important;
  }
  tr.visible-sm {
    display: table-row !important;
  }
  th.visible-sm,
  td.visible-sm {
    display: table-cell !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm-block {
    display: block !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm-inline {
    display: inline !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm-inline-block {
    display: inline-block !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md {
    display: block !important;
  }
  table.visible-md {
    display: table !important;
  }
  tr.visible-md {
    display: table-row !important;
  }
  th.visible-md,
  td.visible-md {
    display: table-cell !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md-block {
    display: block !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md-inline {
    display: inline !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md-inline-block {
    display: inline-block !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg {
    display: block !important;
  }
  table.visible-lg {
    display: table !important;
  }
  tr.visible-lg {
    display: table-row !important;
  }
  th.visible-lg,
  td.visible-lg {
    display: table-cell !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg-block {
    display: block !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg-inline {
    display: inline !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg-inline-block {
    display: inline-block !important;
  }
}
@media (max-width: 767px) {
  .hidden-xs {
    display: none !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .hidden-sm {
    display: none !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .hidden-md {
    display: none !important;
  }
}
@media (min-width: 1200px) {
  .hidden-lg {
    display: none !important;
  }
}
.visible-print {
  display: none !important;
}
@media print {
  .visible-print {
    display: block !important;
  }
  table.visible-print {
    display: table !important;
  }
  tr.visible-print {
    display: table-row !important;
  }
  th.visible-print,
  td.visible-print {
    display: table-cell !important;
  }
}
.visible-print-block {
  display: none !important;
}
@media print {
  .visible-print-block {
    display: block !important;
  }
}
.visible-print-inline {
  display: none !important;
}
@media print {
  .visible-print-inline {
    display: inline !important;
  }
}
.visible-print-inline-block {
  display: none !important;
}
@media print {
  .visible-print-inline-block {
    display: inline-block !important;
  }
}
@media print {
  .hidden-print {
    display: none !important;
  }
}
/*!
*
* Font Awesome
*
*/
/*!
 *  Font Awesome 4.2.0 by @davegandy - http://fontawesome.io - @fontawesome
 *  License - http://fontawesome.io/license (Font: SIL OFL 1.1, CSS: MIT License)
 */
/* FONT PATH
 * -------------------------- */
@font-face {
  font-family: 'FontAwesome';
  src: url('../components/font-awesome/fonts/fontawesome-webfont.eot?v=4.2.0');
  src: url('../components/font-awesome/fonts/fontawesome-webfont.eot?#iefix&v=4.2.0') format('embedded-opentype'), url('../components/font-awesome/fonts/fontawesome-webfont.woff?v=4.2.0') format('woff'), url('../components/font-awesome/fonts/fontawesome-webfont.ttf?v=4.2.0') format('truetype'), url('../components/font-awesome/fonts/fontawesome-webfont.svg?v=4.2.0#fontawesomeregular') format('svg');
  font-weight: normal;
  font-style: normal;
}
.fa {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
/* makes the font 33% larger relative to the icon container */
.fa-lg {
  font-size: 1.33333333em;
  line-height: 0.75em;
  vertical-align: -15%;
}
.fa-2x {
  font-size: 2em;
}
.fa-3x {
  font-size: 3em;
}
.fa-4x {
  font-size: 4em;
}
.fa-5x {
  font-size: 5em;
}
.fa-fw {
  width: 1.28571429em;
  text-align: center;
}
.fa-ul {
  padding-left: 0;
  margin-left: 2.14285714em;
  list-style-type: none;
}
.fa-ul > li {
  position: relative;
}
.fa-li {
  position: absolute;
  left: -2.14285714em;
  width: 2.14285714em;
  top: 0.14285714em;
  text-align: center;
}
.fa-li.fa-lg {
  left: -1.85714286em;
}
.fa-border {
  padding: .2em .25em .15em;
  border: solid 0.08em #eee;
  border-radius: .1em;
}
.pull-right {
  float: right;
}
.pull-left {
  float: left;
}
.fa.pull-left {
  margin-right: .3em;
}
.fa.pull-right {
  margin-left: .3em;
}
.fa-spin {
  -webkit-animation: fa-spin 2s infinite linear;
  animation: fa-spin 2s infinite linear;
}
@-webkit-keyframes fa-spin {
  0% {
    -webkit-transform: rotate(0deg);
    transform: rotate(0deg);
  }
  100% {
    -webkit-transform: rotate(359deg);
    transform: rotate(359deg);
  }
}
@keyframes fa-spin {
  0% {
    -webkit-transform: rotate(0deg);
    transform: rotate(0deg);
  }
  100% {
    -webkit-transform: rotate(359deg);
    transform: rotate(359deg);
  }
}
.fa-rotate-90 {
  filter: progid:DXImageTransform.Microsoft.BasicImage(rotation=1);
  -webkit-transform: rotate(90deg);
  -ms-transform: rotate(90deg);
  transform: rotate(90deg);
}
.fa-rotate-180 {
  filter: progid:DXImageTransform.Microsoft.BasicImage(rotation=2);
  -webkit-transform: rotate(180deg);
  -ms-transform: rotate(180deg);
  transform: rotate(180deg);
}
.fa-rotate-270 {
  filter: progid:DXImageTransform.Microsoft.BasicImage(rotation=3);
  -webkit-transform: rotate(270deg);
  -ms-transform: rotate(270deg);
  transform: rotate(270deg);
}
.fa-flip-horizontal {
  filter: progid:DXImageTransform.Microsoft.BasicImage(rotation=0, mirror=1);
  -webkit-transform: scale(-1, 1);
  -ms-transform: scale(-1, 1);
  transform: scale(-1, 1);
}
.fa-flip-vertical {
  filter: progid:DXImageTransform.Microsoft.BasicImage(rotation=2, mirror=1);
  -webkit-transform: scale(1, -1);
  -ms-transform: scale(1, -1);
  transform: scale(1, -1);
}
:root .fa-rotate-90,
:root .fa-rotate-180,
:root .fa-rotate-270,
:root .fa-flip-horizontal,
:root .fa-flip-vertical {
  filter: none;
}
.fa-stack {
  position: relative;
  display: inline-block;
  width: 2em;
  height: 2em;
  line-height: 2em;
  vertical-align: middle;
}
.fa-stack-1x,
.fa-stack-2x {
  position: absolute;
  left: 0;
  width: 100%;
  text-align: center;
}
.fa-stack-1x {
  line-height: inherit;
}
.fa-stack-2x {
  font-size: 2em;
}
.fa-inverse {
  color: #fff;
}
/* Font Awesome uses the Unicode Private Use Area (PUA) to ensure screen
   readers do not read off random characters that represent icons */
.fa-glass:before {
  content: "\f000";
}
.fa-music:before {
  content: "\f001";
}
.fa-search:before {
  content: "\f002";
}
.fa-envelope-o:before {
  content: "\f003";
}
.fa-heart:before {
  content: "\f004";
}
.fa-star:before {
  content: "\f005";
}
.fa-star-o:before {
  content: "\f006";
}
.fa-user:before {
  content: "\f007";
}
.fa-film:before {
  content: "\f008";
}
.fa-th-large:before {
  content: "\f009";
}
.fa-th:before {
  content: "\f00a";
}
.fa-th-list:before {
  content: "\f00b";
}
.fa-check:before {
  content: "\f00c";
}
.fa-remove:before,
.fa-close:before,
.fa-times:before {
  content: "\f00d";
}
.fa-search-plus:before {
  content: "\f00e";
}
.fa-search-minus:before {
  content: "\f010";
}
.fa-power-off:before {
  content: "\f011";
}
.fa-signal:before {
  content: "\f012";
}
.fa-gear:before,
.fa-cog:before {
  content: "\f013";
}
.fa-trash-o:before {
  content: "\f014";
}
.fa-home:before {
  content: "\f015";
}
.fa-file-o:before {
  content: "\f016";
}
.fa-clock-o:before {
  content: "\f017";
}
.fa-road:before {
  content: "\f018";
}
.fa-download:before {
  content: "\f019";
}
.fa-arrow-circle-o-down:before {
  content: "\f01a";
}
.fa-arrow-circle-o-up:before {
  content: "\f01b";
}
.fa-inbox:before {
  content: "\f01c";
}
.fa-play-circle-o:before {
  content: "\f01d";
}
.fa-rotate-right:before,
.fa-repeat:before {
  content: "\f01e";
}
.fa-refresh:before {
  content: "\f021";
}
.fa-list-alt:before {
  content: "\f022";
}
.fa-lock:before {
  content: "\f023";
}
.fa-flag:before {
  content: "\f024";
}
.fa-headphones:before {
  content: "\f025";
}
.fa-volume-off:before {
  content: "\f026";
}
.fa-volume-down:before {
  content: "\f027";
}
.fa-volume-up:before {
  content: "\f028";
}
.fa-qrcode:before {
  content: "\f029";
}
.fa-barcode:before {
  content: "\f02a";
}
.fa-tag:before {
  content: "\f02b";
}
.fa-tags:before {
  content: "\f02c";
}
.fa-book:before {
  content: "\f02d";
}
.fa-bookmark:before {
  content: "\f02e";
}
.fa-print:before {
  content: "\f02f";
}
.fa-camera:before {
  content: "\f030";
}
.fa-font:before {
  content: "\f031";
}
.fa-bold:before {
  content: "\f032";
}
.fa-italic:before {
  content: "\f033";
}
.fa-text-height:before {
  content: "\f034";
}
.fa-text-width:before {
  content: "\f035";
}
.fa-align-left:before {
  content: "\f036";
}
.fa-align-center:before {
  content: "\f037";
}
.fa-align-right:before {
  content: "\f038";
}
.fa-align-justify:before {
  content: "\f039";
}
.fa-list:before {
  content: "\f03a";
}
.fa-dedent:before,
.fa-outdent:before {
  content: "\f03b";
}
.fa-indent:before {
  content: "\f03c";
}
.fa-video-camera:before {
  content: "\f03d";
}
.fa-photo:before,
.fa-image:before,
.fa-picture-o:before {
  content: "\f03e";
}
.fa-pencil:before {
  content: "\f040";
}
.fa-map-marker:before {
  content: "\f041";
}
.fa-adjust:before {
  content: "\f042";
}
.fa-tint:before {
  content: "\f043";
}
.fa-edit:before,
.fa-pencil-square-o:before {
  content: "\f044";
}
.fa-share-square-o:before {
  content: "\f045";
}
.fa-check-square-o:before {
  content: "\f046";
}
.fa-arrows:before {
  content: "\f047";
}
.fa-step-backward:before {
  content: "\f048";
}
.fa-fast-backward:before {
  content: "\f049";
}
.fa-backward:before {
  content: "\f04a";
}
.fa-play:before {
  content: "\f04b";
}
.fa-pause:before {
  content: "\f04c";
}
.fa-stop:before {
  content: "\f04d";
}
.fa-forward:before {
  content: "\f04e";
}
.fa-fast-forward:before {
  content: "\f050";
}
.fa-step-forward:before {
  content: "\f051";
}
.fa-eject:before {
  content: "\f052";
}
.fa-chevron-left:before {
  content: "\f053";
}
.fa-chevron-right:before {
  content: "\f054";
}
.fa-plus-circle:before {
  content: "\f055";
}
.fa-minus-circle:before {
  content: "\f056";
}
.fa-times-circle:before {
  content: "\f057";
}
.fa-check-circle:before {
  content: "\f058";
}
.fa-question-circle:before {
  content: "\f059";
}
.fa-info-circle:before {
  content: "\f05a";
}
.fa-crosshairs:before {
  content: "\f05b";
}
.fa-times-circle-o:before {
  content: "\f05c";
}
.fa-check-circle-o:before {
  content: "\f05d";
}
.fa-ban:before {
  content: "\f05e";
}
.fa-arrow-left:before {
  content: "\f060";
}
.fa-arrow-right:before {
  content: "\f061";
}
.fa-arrow-up:before {
  content: "\f062";
}
.fa-arrow-down:before {
  content: "\f063";
}
.fa-mail-forward:before,
.fa-share:before {
  content: "\f064";
}
.fa-expand:before {
  content: "\f065";
}
.fa-compress:before {
  content: "\f066";
}
.fa-plus:before {
  content: "\f067";
}
.fa-minus:before {
  content: "\f068";
}
.fa-asterisk:before {
  content: "\f069";
}
.fa-exclamation-circle:before {
  content: "\f06a";
}
.fa-gift:before {
  content: "\f06b";
}
.fa-leaf:before {
  content: "\f06c";
}
.fa-fire:before {
  content: "\f06d";
}
.fa-eye:before {
  content: "\f06e";
}
.fa-eye-slash:before {
  content: "\f070";
}
.fa-warning:before,
.fa-exclamation-triangle:before {
  content: "\f071";
}
.fa-plane:before {
  content: "\f072";
}
.fa-calendar:before {
  content: "\f073";
}
.fa-random:before {
  content: "\f074";
}
.fa-comment:before {
  content: "\f075";
}
.fa-magnet:before {
  content: "\f076";
}
.fa-chevron-up:before {
  content: "\f077";
}
.fa-chevron-down:before {
  content: "\f078";
}
.fa-retweet:before {
  content: "\f079";
}
.fa-shopping-cart:before {
  content: "\f07a";
}
.fa-folder:before {
  content: "\f07b";
}
.fa-folder-open:before {
  content: "\f07c";
}
.fa-arrows-v:before {
  content: "\f07d";
}
.fa-arrows-h:before {
  content: "\f07e";
}
.fa-bar-chart-o:before,
.fa-bar-chart:before {
  content: "\f080";
}
.fa-twitter-square:before {
  content: "\f081";
}
.fa-facebook-square:before {
  content: "\f082";
}
.fa-camera-retro:before {
  content: "\f083";
}
.fa-key:before {
  content: "\f084";
}
.fa-gears:before,
.fa-cogs:before {
  content: "\f085";
}
.fa-comments:before {
  content: "\f086";
}
.fa-thumbs-o-up:before {
  content: "\f087";
}
.fa-thumbs-o-down:before {
  content: "\f088";
}
.fa-star-half:before {
  content: "\f089";
}
.fa-heart-o:before {
  content: "\f08a";
}
.fa-sign-out:before {
  content: "\f08b";
}
.fa-linkedin-square:before {
  content: "\f08c";
}
.fa-thumb-tack:before {
  content: "\f08d";
}
.fa-external-link:before {
  content: "\f08e";
}
.fa-sign-in:before {
  content: "\f090";
}
.fa-trophy:before {
  content: "\f091";
}
.fa-github-square:before {
  content: "\f092";
}
.fa-upload:before {
  content: "\f093";
}
.fa-lemon-o:before {
  content: "\f094";
}
.fa-phone:before {
  content: "\f095";
}
.fa-square-o:before {
  content: "\f096";
}
.fa-bookmark-o:before {
  content: "\f097";
}
.fa-phone-square:before {
  content: "\f098";
}
.fa-twitter:before {
  content: "\f099";
}
.fa-facebook:before {
  content: "\f09a";
}
.fa-github:before {
  content: "\f09b";
}
.fa-unlock:before {
  content: "\f09c";
}
.fa-credit-card:before {
  content: "\f09d";
}
.fa-rss:before {
  content: "\f09e";
}
.fa-hdd-o:before {
  content: "\f0a0";
}
.fa-bullhorn:before {
  content: "\f0a1";
}
.fa-bell:before {
  content: "\f0f3";
}
.fa-certificate:before {
  content: "\f0a3";
}
.fa-hand-o-right:before {
  content: "\f0a4";
}
.fa-hand-o-left:before {
  content: "\f0a5";
}
.fa-hand-o-up:before {
  content: "\f0a6";
}
.fa-hand-o-down:before {
  content: "\f0a7";
}
.fa-arrow-circle-left:before {
  content: "\f0a8";
}
.fa-arrow-circle-right:before {
  content: "\f0a9";
}
.fa-arrow-circle-up:before {
  content: "\f0aa";
}
.fa-arrow-circle-down:before {
  content: "\f0ab";
}
.fa-globe:before {
  content: "\f0ac";
}
.fa-wrench:before {
  content: "\f0ad";
}
.fa-tasks:before {
  content: "\f0ae";
}
.fa-filter:before {
  content: "\f0b0";
}
.fa-briefcase:before {
  content: "\f0b1";
}
.fa-arrows-alt:before {
  content: "\f0b2";
}
.fa-group:before,
.fa-users:before {
  content: "\f0c0";
}
.fa-chain:before,
.fa-link:before {
  content: "\f0c1";
}
.fa-cloud:before {
  content: "\f0c2";
}
.fa-flask:before {
  content: "\f0c3";
}
.fa-cut:before,
.fa-scissors:before {
  content: "\f0c4";
}
.fa-copy:before,
.fa-files-o:before {
  content: "\f0c5";
}
.fa-paperclip:before {
  content: "\f0c6";
}
.fa-save:before,
.fa-floppy-o:before {
  content: "\f0c7";
}
.fa-square:before {
  content: "\f0c8";
}
.fa-navicon:before,
.fa-reorder:before,
.fa-bars:before {
  content: "\f0c9";
}
.fa-list-ul:before {
  content: "\f0ca";
}
.fa-list-ol:before {
  content: "\f0cb";
}
.fa-strikethrough:before {
  content: "\f0cc";
}
.fa-underline:before {
  content: "\f0cd";
}
.fa-table:before {
  content: "\f0ce";
}
.fa-magic:before {
  content: "\f0d0";
}
.fa-truck:before {
  content: "\f0d1";
}
.fa-pinterest:before {
  content: "\f0d2";
}
.fa-pinterest-square:before {
  content: "\f0d3";
}
.fa-google-plus-square:before {
  content: "\f0d4";
}
.fa-google-plus:before {
  content: "\f0d5";
}
.fa-money:before {
  content: "\f0d6";
}
.fa-caret-down:before {
  content: "\f0d7";
}
.fa-caret-up:before {
  content: "\f0d8";
}
.fa-caret-left:before {
  content: "\f0d9";
}
.fa-caret-right:before {
  content: "\f0da";
}
.fa-columns:before {
  content: "\f0db";
}
.fa-unsorted:before,
.fa-sort:before {
  content: "\f0dc";
}
.fa-sort-down:before,
.fa-sort-desc:before {
  content: "\f0dd";
}
.fa-sort-up:before,
.fa-sort-asc:before {
  content: "\f0de";
}
.fa-envelope:before {
  content: "\f0e0";
}
.fa-linkedin:before {
  content: "\f0e1";
}
.fa-rotate-left:before,
.fa-undo:before {
  content: "\f0e2";
}
.fa-legal:before,
.fa-gavel:before {
  content: "\f0e3";
}
.fa-dashboard:before,
.fa-tachometer:before {
  content: "\f0e4";
}
.fa-comment-o:before {
  content: "\f0e5";
}
.fa-comments-o:before {
  content: "\f0e6";
}
.fa-flash:before,
.fa-bolt:before {
  content: "\f0e7";
}
.fa-sitemap:before {
  content: "\f0e8";
}
.fa-umbrella:before {
  content: "\f0e9";
}
.fa-paste:before,
.fa-clipboard:before {
  content: "\f0ea";
}
.fa-lightbulb-o:before {
  content: "\f0eb";
}
.fa-exchange:before {
  content: "\f0ec";
}
.fa-cloud-download:before {
  content: "\f0ed";
}
.fa-cloud-upload:before {
  content: "\f0ee";
}
.fa-user-md:before {
  content: "\f0f0";
}
.fa-stethoscope:before {
  content: "\f0f1";
}
.fa-suitcase:before {
  content: "\f0f2";
}
.fa-bell-o:before {
  content: "\f0a2";
}
.fa-coffee:before {
  content: "\f0f4";
}
.fa-cutlery:before {
  content: "\f0f5";
}
.fa-file-text-o:before {
  content: "\f0f6";
}
.fa-building-o:before {
  content: "\f0f7";
}
.fa-hospital-o:before {
  content: "\f0f8";
}
.fa-ambulance:before {
  content: "\f0f9";
}
.fa-medkit:before {
  content: "\f0fa";
}
.fa-fighter-jet:before {
  content: "\f0fb";
}
.fa-beer:before {
  content: "\f0fc";
}
.fa-h-square:before {
  content: "\f0fd";
}
.fa-plus-square:before {
  content: "\f0fe";
}
.fa-angle-double-left:before {
  content: "\f100";
}
.fa-angle-double-right:before {
  content: "\f101";
}
.fa-angle-double-up:before {
  content: "\f102";
}
.fa-angle-double-down:before {
  content: "\f103";
}
.fa-angle-left:before {
  content: "\f104";
}
.fa-angle-right:before {
  content: "\f105";
}
.fa-angle-up:before {
  content: "\f106";
}
.fa-angle-down:before {
  content: "\f107";
}
.fa-desktop:before {
  content: "\f108";
}
.fa-laptop:before {
  content: "\f109";
}
.fa-tablet:before {
  content: "\f10a";
}
.fa-mobile-phone:before,
.fa-mobile:before {
  content: "\f10b";
}
.fa-circle-o:before {
  content: "\f10c";
}
.fa-quote-left:before {
  content: "\f10d";
}
.fa-quote-right:before {
  content: "\f10e";
}
.fa-spinner:before {
  content: "\f110";
}
.fa-circle:before {
  content: "\f111";
}
.fa-mail-reply:before,
.fa-reply:before {
  content: "\f112";
}
.fa-github-alt:before {
  content: "\f113";
}
.fa-folder-o:before {
  content: "\f114";
}
.fa-folder-open-o:before {
  content: "\f115";
}
.fa-smile-o:before {
  content: "\f118";
}
.fa-frown-o:before {
  content: "\f119";
}
.fa-meh-o:before {
  content: "\f11a";
}
.fa-gamepad:before {
  content: "\f11b";
}
.fa-keyboard-o:before {
  content: "\f11c";
}
.fa-flag-o:before {
  content: "\f11d";
}
.fa-flag-checkered:before {
  content: "\f11e";
}
.fa-terminal:before {
  content: "\f120";
}
.fa-code:before {
  content: "\f121";
}
.fa-mail-reply-all:before,
.fa-reply-all:before {
  content: "\f122";
}
.fa-star-half-empty:before,
.fa-star-half-full:before,
.fa-star-half-o:before {
  content: "\f123";
}
.fa-location-arrow:before {
  content: "\f124";
}
.fa-crop:before {
  content: "\f125";
}
.fa-code-fork:before {
  content: "\f126";
}
.fa-unlink:before,
.fa-chain-broken:before {
  content: "\f127";
}
.fa-question:before {
  content: "\f128";
}
.fa-info:before {
  content: "\f129";
}
.fa-exclamation:before {
  content: "\f12a";
}
.fa-superscript:before {
  content: "\f12b";
}
.fa-subscript:before {
  content: "\f12c";
}
.fa-eraser:before {
  content: "\f12d";
}
.fa-puzzle-piece:before {
  content: "\f12e";
}
.fa-microphone:before {
  content: "\f130";
}
.fa-microphone-slash:before {
  content: "\f131";
}
.fa-shield:before {
  content: "\f132";
}
.fa-calendar-o:before {
  content: "\f133";
}
.fa-fire-extinguisher:before {
  content: "\f134";
}
.fa-rocket:before {
  content: "\f135";
}
.fa-maxcdn:before {
  content: "\f136";
}
.fa-chevron-circle-left:before {
  content: "\f137";
}
.fa-chevron-circle-right:before {
  content: "\f138";
}
.fa-chevron-circle-up:before {
  content: "\f139";
}
.fa-chevron-circle-down:before {
  content: "\f13a";
}
.fa-html5:before {
  content: "\f13b";
}
.fa-css3:before {
  content: "\f13c";
}
.fa-anchor:before {
  content: "\f13d";
}
.fa-unlock-alt:before {
  content: "\f13e";
}
.fa-bullseye:before {
  content: "\f140";
}
.fa-ellipsis-h:before {
  content: "\f141";
}
.fa-ellipsis-v:before {
  content: "\f142";
}
.fa-rss-square:before {
  content: "\f143";
}
.fa-play-circle:before {
  content: "\f144";
}
.fa-ticket:before {
  content: "\f145";
}
.fa-minus-square:before {
  content: "\f146";
}
.fa-minus-square-o:before {
  content: "\f147";
}
.fa-level-up:before {
  content: "\f148";
}
.fa-level-down:before {
  content: "\f149";
}
.fa-check-square:before {
  content: "\f14a";
}
.fa-pencil-square:before {
  content: "\f14b";
}
.fa-external-link-square:before {
  content: "\f14c";
}
.fa-share-square:before {
  content: "\f14d";
}
.fa-compass:before {
  content: "\f14e";
}
.fa-toggle-down:before,
.fa-caret-square-o-down:before {
  content: "\f150";
}
.fa-toggle-up:before,
.fa-caret-square-o-up:before {
  content: "\f151";
}
.fa-toggle-right:before,
.fa-caret-square-o-right:before {
  content: "\f152";
}
.fa-euro:before,
.fa-eur:before {
  content: "\f153";
}
.fa-gbp:before {
  content: "\f154";
}
.fa-dollar:before,
.fa-usd:before {
  content: "\f155";
}
.fa-rupee:before,
.fa-inr:before {
  content: "\f156";
}
.fa-cny:before,
.fa-rmb:before,
.fa-yen:before,
.fa-jpy:before {
  content: "\f157";
}
.fa-ruble:before,
.fa-rouble:before,
.fa-rub:before {
  content: "\f158";
}
.fa-won:before,
.fa-krw:before {
  content: "\f159";
}
.fa-bitcoin:before,
.fa-btc:before {
  content: "\f15a";
}
.fa-file:before {
  content: "\f15b";
}
.fa-file-text:before {
  content: "\f15c";
}
.fa-sort-alpha-asc:before {
  content: "\f15d";
}
.fa-sort-alpha-desc:before {
  content: "\f15e";
}
.fa-sort-amount-asc:before {
  content: "\f160";
}
.fa-sort-amount-desc:before {
  content: "\f161";
}
.fa-sort-numeric-asc:before {
  content: "\f162";
}
.fa-sort-numeric-desc:before {
  content: "\f163";
}
.fa-thumbs-up:before {
  content: "\f164";
}
.fa-thumbs-down:before {
  content: "\f165";
}
.fa-youtube-square:before {
  content: "\f166";
}
.fa-youtube:before {
  content: "\f167";
}
.fa-xing:before {
  content: "\f168";
}
.fa-xing-square:before {
  content: "\f169";
}
.fa-youtube-play:before {
  content: "\f16a";
}
.fa-dropbox:before {
  content: "\f16b";
}
.fa-stack-overflow:before {
  content: "\f16c";
}
.fa-instagram:before {
  content: "\f16d";
}
.fa-flickr:before {
  content: "\f16e";
}
.fa-adn:before {
  content: "\f170";
}
.fa-bitbucket:before {
  content: "\f171";
}
.fa-bitbucket-square:before {
  content: "\f172";
}
.fa-tumblr:before {
  content: "\f173";
}
.fa-tumblr-square:before {
  content: "\f174";
}
.fa-long-arrow-down:before {
  content: "\f175";
}
.fa-long-arrow-up:before {
  content: "\f176";
}
.fa-long-arrow-left:before {
  content: "\f177";
}
.fa-long-arrow-right:before {
  content: "\f178";
}
.fa-apple:before {
  content: "\f179";
}
.fa-windows:before {
  content: "\f17a";
}
.fa-android:before {
  content: "\f17b";
}
.fa-linux:before {
  content: "\f17c";
}
.fa-dribbble:before {
  content: "\f17d";
}
.fa-skype:before {
  content: "\f17e";
}
.fa-foursquare:before {
  content: "\f180";
}
.fa-trello:before {
  content: "\f181";
}
.fa-female:before {
  content: "\f182";
}
.fa-male:before {
  content: "\f183";
}
.fa-gittip:before {
  content: "\f184";
}
.fa-sun-o:before {
  content: "\f185";
}
.fa-moon-o:before {
  content: "\f186";
}
.fa-archive:before {
  content: "\f187";
}
.fa-bug:before {
  content: "\f188";
}
.fa-vk:before {
  content: "\f189";
}
.fa-weibo:before {
  content: "\f18a";
}
.fa-renren:before {
  content: "\f18b";
}
.fa-pagelines:before {
  content: "\f18c";
}
.fa-stack-exchange:before {
  content: "\f18d";
}
.fa-arrow-circle-o-right:before {
  content: "\f18e";
}
.fa-arrow-circle-o-left:before {
  content: "\f190";
}
.fa-toggle-left:before,
.fa-caret-square-o-left:before {
  content: "\f191";
}
.fa-dot-circle-o:before {
  content: "\f192";
}
.fa-wheelchair:before {
  content: "\f193";
}
.fa-vimeo-square:before {
  content: "\f194";
}
.fa-turkish-lira:before,
.fa-try:before {
  content: "\f195";
}
.fa-plus-square-o:before {
  content: "\f196";
}
.fa-space-shuttle:before {
  content: "\f197";
}
.fa-slack:before {
  content: "\f198";
}
.fa-envelope-square:before {
  content: "\f199";
}
.fa-wordpress:before {
  content: "\f19a";
}
.fa-openid:before {
  content: "\f19b";
}
.fa-institution:before,
.fa-bank:before,
.fa-university:before {
  content: "\f19c";
}
.fa-mortar-board:before,
.fa-graduation-cap:before {
  content: "\f19d";
}
.fa-yahoo:before {
  content: "\f19e";
}
.fa-google:before {
  content: "\f1a0";
}
.fa-reddit:before {
  content: "\f1a1";
}
.fa-reddit-square:before {
  content: "\f1a2";
}
.fa-stumbleupon-circle:before {
  content: "\f1a3";
}
.fa-stumbleupon:before {
  content: "\f1a4";
}
.fa-delicious:before {
  content: "\f1a5";
}
.fa-digg:before {
  content: "\f1a6";
}
.fa-pied-piper:before {
  content: "\f1a7";
}
.fa-pied-piper-alt:before {
  content: "\f1a8";
}
.fa-drupal:before {
  content: "\f1a9";
}
.fa-joomla:before {
  content: "\f1aa";
}
.fa-language:before {
  content: "\f1ab";
}
.fa-fax:before {
  content: "\f1ac";
}
.fa-building:before {
  content: "\f1ad";
}
.fa-child:before {
  content: "\f1ae";
}
.fa-paw:before {
  content: "\f1b0";
}
.fa-spoon:before {
  content: "\f1b1";
}
.fa-cube:before {
  content: "\f1b2";
}
.fa-cubes:before {
  content: "\f1b3";
}
.fa-behance:before {
  content: "\f1b4";
}
.fa-behance-square:before {
  content: "\f1b5";
}
.fa-steam:before {
  content: "\f1b6";
}
.fa-steam-square:before {
  content: "\f1b7";
}
.fa-recycle:before {
  content: "\f1b8";
}
.fa-automobile:before,
.fa-car:before {
  content: "\f1b9";
}
.fa-cab:before,
.fa-taxi:before {
  content: "\f1ba";
}
.fa-tree:before {
  content: "\f1bb";
}
.fa-spotify:before {
  content: "\f1bc";
}
.fa-deviantart:before {
  content: "\f1bd";
}
.fa-soundcloud:before {
  content: "\f1be";
}
.fa-database:before {
  content: "\f1c0";
}
.fa-file-pdf-o:before {
  content: "\f1c1";
}
.fa-file-word-o:before {
  content: "\f1c2";
}
.fa-file-excel-o:before {
  content: "\f1c3";
}
.fa-file-powerpoint-o:before {
  content: "\f1c4";
}
.fa-file-photo-o:before,
.fa-file-picture-o:before,
.fa-file-image-o:before {
  content: "\f1c5";
}
.fa-file-zip-o:before,
.fa-file-archive-o:before {
  content: "\f1c6";
}
.fa-file-sound-o:before,
.fa-file-audio-o:before {
  content: "\f1c7";
}
.fa-file-movie-o:before,
.fa-file-video-o:before {
  content: "\f1c8";
}
.fa-file-code-o:before {
  content: "\f1c9";
}
.fa-vine:before {
  content: "\f1ca";
}
.fa-codepen:before {
  content: "\f1cb";
}
.fa-jsfiddle:before {
  content: "\f1cc";
}
.fa-life-bouy:before,
.fa-life-buoy:before,
.fa-life-saver:before,
.fa-support:before,
.fa-life-ring:before {
  content: "\f1cd";
}
.fa-circle-o-notch:before {
  content: "\f1ce";
}
.fa-ra:before,
.fa-rebel:before {
  content: "\f1d0";
}
.fa-ge:before,
.fa-empire:before {
  content: "\f1d1";
}
.fa-git-square:before {
  content: "\f1d2";
}
.fa-git:before {
  content: "\f1d3";
}
.fa-hacker-news:before {
  content: "\f1d4";
}
.fa-tencent-weibo:before {
  content: "\f1d5";
}
.fa-qq:before {
  content: "\f1d6";
}
.fa-wechat:before,
.fa-weixin:before {
  content: "\f1d7";
}
.fa-send:before,
.fa-paper-plane:before {
  content: "\f1d8";
}
.fa-send-o:before,
.fa-paper-plane-o:before {
  content: "\f1d9";
}
.fa-history:before {
  content: "\f1da";
}
.fa-circle-thin:before {
  content: "\f1db";
}
.fa-header:before {
  content: "\f1dc";
}
.fa-paragraph:before {
  content: "\f1dd";
}
.fa-sliders:before {
  content: "\f1de";
}
.fa-share-alt:before {
  content: "\f1e0";
}
.fa-share-alt-square:before {
  content: "\f1e1";
}
.fa-bomb:before {
  content: "\f1e2";
}
.fa-soccer-ball-o:before,
.fa-futbol-o:before {
  content: "\f1e3";
}
.fa-tty:before {
  content: "\f1e4";
}
.fa-binoculars:before {
  content: "\f1e5";
}
.fa-plug:before {
  content: "\f1e6";
}
.fa-slideshare:before {
  content: "\f1e7";
}
.fa-twitch:before {
  content: "\f1e8";
}
.fa-yelp:before {
  content: "\f1e9";
}
.fa-newspaper-o:before {
  content: "\f1ea";
}
.fa-wifi:before {
  content: "\f1eb";
}
.fa-calculator:before {
  content: "\f1ec";
}
.fa-paypal:before {
  content: "\f1ed";
}
.fa-google-wallet:before {
  content: "\f1ee";
}
.fa-cc-visa:before {
  content: "\f1f0";
}
.fa-cc-mastercard:before {
  content: "\f1f1";
}
.fa-cc-discover:before {
  content: "\f1f2";
}
.fa-cc-amex:before {
  content: "\f1f3";
}
.fa-cc-paypal:before {
  content: "\f1f4";
}
.fa-cc-stripe:before {
  content: "\f1f5";
}
.fa-bell-slash:before {
  content: "\f1f6";
}
.fa-bell-slash-o:before {
  content: "\f1f7";
}
.fa-trash:before {
  content: "\f1f8";
}
.fa-copyright:before {
  content: "\f1f9";
}
.fa-at:before {
  content: "\f1fa";
}
.fa-eyedropper:before {
  content: "\f1fb";
}
.fa-paint-brush:before {
  content: "\f1fc";
}
.fa-birthday-cake:before {
  content: "\f1fd";
}
.fa-area-chart:before {
  content: "\f1fe";
}
.fa-pie-chart:before {
  content: "\f200";
}
.fa-line-chart:before {
  content: "\f201";
}
.fa-lastfm:before {
  content: "\f202";
}
.fa-lastfm-square:before {
  content: "\f203";
}
.fa-toggle-off:before {
  content: "\f204";
}
.fa-toggle-on:before {
  content: "\f205";
}
.fa-bicycle:before {
  content: "\f206";
}
.fa-bus:before {
  content: "\f207";
}
.fa-ioxhost:before {
  content: "\f208";
}
.fa-angellist:before {
  content: "\f209";
}
.fa-cc:before {
  content: "\f20a";
}
.fa-shekel:before,
.fa-sheqel:before,
.fa-ils:before {
  content: "\f20b";
}
.fa-meanpath:before {
  content: "\f20c";
}
/*!
*
* IPython base
*
*/
.modal.fade .modal-dialog {
  -webkit-transform: translate(0, 0);
  -ms-transform: translate(0, 0);
  -o-transform: translate(0, 0);
  transform: translate(0, 0);
}
code {
  color: #000;
}
pre {
  font-size: inherit;
  line-height: inherit;
}
label {
  font-weight: normal;
}
/* Make the page background atleast 100% the height of the view port */
/* Make the page itself atleast 70% the height of the view port */
.border-box-sizing {
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
.corner-all {
  border-radius: 2px;
}
.no-padding {
  padding: 0px;
}
/* Flexible box model classes */
/* Taken from Alex Russell http://infrequently.org/2009/08/css-3-progress/ */
/* This file is a compatability layer.  It allows the usage of flexible box 
model layouts accross multiple browsers, including older browsers.  The newest,
universal implementation of the flexible box model is used when available (see
`Modern browsers` comments below).  Browsers that are known to implement this 
new spec completely include:

    Firefox 28.0+
    Chrome 29.0+
    Internet Explorer 11+ 
    Opera 17.0+

Browsers not listed, including Safari, are supported via the styling under the
`Old browsers` comments below.
*/
.hbox {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
.hbox > * {
  /* Old browsers */
  -webkit-box-flex: 0;
  -moz-box-flex: 0;
  box-flex: 0;
  /* Modern browsers */
  flex: none;
}
.vbox {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
.vbox > * {
  /* Old browsers */
  -webkit-box-flex: 0;
  -moz-box-flex: 0;
  box-flex: 0;
  /* Modern browsers */
  flex: none;
}
.hbox.reverse,
.vbox.reverse,
.reverse {
  /* Old browsers */
  -webkit-box-direction: reverse;
  -moz-box-direction: reverse;
  box-direction: reverse;
  /* Modern browsers */
  flex-direction: row-reverse;
}
.hbox.box-flex0,
.vbox.box-flex0,
.box-flex0 {
  /* Old browsers */
  -webkit-box-flex: 0;
  -moz-box-flex: 0;
  box-flex: 0;
  /* Modern browsers */
  flex: none;
  width: auto;
}
.hbox.box-flex1,
.vbox.box-flex1,
.box-flex1 {
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
.hbox.box-flex,
.vbox.box-flex,
.box-flex {
  /* Old browsers */
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
.hbox.box-flex2,
.vbox.box-flex2,
.box-flex2 {
  /* Old browsers */
  -webkit-box-flex: 2;
  -moz-box-flex: 2;
  box-flex: 2;
  /* Modern browsers */
  flex: 2;
}
.box-group1 {
  /*  Deprecated */
  -webkit-box-flex-group: 1;
  -moz-box-flex-group: 1;
  box-flex-group: 1;
}
.box-group2 {
  /* Deprecated */
  -webkit-box-flex-group: 2;
  -moz-box-flex-group: 2;
  box-flex-group: 2;
}
.hbox.start,
.vbox.start,
.start {
  /* Old browsers */
  -webkit-box-pack: start;
  -moz-box-pack: start;
  box-pack: start;
  /* Modern browsers */
  justify-content: flex-start;
}
.hbox.end,
.vbox.end,
.end {
  /* Old browsers */
  -webkit-box-pack: end;
  -moz-box-pack: end;
  box-pack: end;
  /* Modern browsers */
  justify-content: flex-end;
}
.hbox.center,
.vbox.center,
.center {
  /* Old browsers */
  -webkit-box-pack: center;
  -moz-box-pack: center;
  box-pack: center;
  /* Modern browsers */
  justify-content: center;
}
.hbox.baseline,
.vbox.baseline,
.baseline {
  /* Old browsers */
  -webkit-box-pack: baseline;
  -moz-box-pack: baseline;
  box-pack: baseline;
  /* Modern browsers */
  justify-content: baseline;
}
.hbox.stretch,
.vbox.stretch,
.stretch {
  /* Old browsers */
  -webkit-box-pack: stretch;
  -moz-box-pack: stretch;
  box-pack: stretch;
  /* Modern browsers */
  justify-content: stretch;
}
.hbox.align-start,
.vbox.align-start,
.align-start {
  /* Old browsers */
  -webkit-box-align: start;
  -moz-box-align: start;
  box-align: start;
  /* Modern browsers */
  align-items: flex-start;
}
.hbox.align-end,
.vbox.align-end,
.align-end {
  /* Old browsers */
  -webkit-box-align: end;
  -moz-box-align: end;
  box-align: end;
  /* Modern browsers */
  align-items: flex-end;
}
.hbox.align-center,
.vbox.align-center,
.align-center {
  /* Old browsers */
  -webkit-box-align: center;
  -moz-box-align: center;
  box-align: center;
  /* Modern browsers */
  align-items: center;
}
.hbox.align-baseline,
.vbox.align-baseline,
.align-baseline {
  /* Old browsers */
  -webkit-box-align: baseline;
  -moz-box-align: baseline;
  box-align: baseline;
  /* Modern browsers */
  align-items: baseline;
}
.hbox.align-stretch,
.vbox.align-stretch,
.align-stretch {
  /* Old browsers */
  -webkit-box-align: stretch;
  -moz-box-align: stretch;
  box-align: stretch;
  /* Modern browsers */
  align-items: stretch;
}
div.error {
  margin: 2em;
  text-align: center;
}
div.error > h1 {
  font-size: 500%;
  line-height: normal;
}
div.error > p {
  font-size: 200%;
  line-height: normal;
}
div.traceback-wrapper {
  text-align: left;
  max-width: 800px;
  margin: auto;
}
/**
 * Primary styles
 *
 * Author: Jupyter Development Team
 */
body {
  background-color: #fff;
  /* This makes sure that the body covers the entire window and needs to
       be in a different element than the display: box in wrapper below */
  position: absolute;
  left: 0px;
  right: 0px;
  top: 0px;
  bottom: 0px;
  overflow: visible;
}
body > #header {
  /* Initially hidden to prevent FLOUC */
  display: none;
  background-color: #fff;
  /* Display over codemirror */
  position: relative;
  z-index: 100;
}
body > #header #header-container {
  padding-bottom: 5px;
  padding-top: 5px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
body > #header .header-bar {
  width: 100%;
  height: 1px;
  background: #e7e7e7;
  margin-bottom: -1px;
}
@media print {
  body > #header {
    display: none !important;
  }
}
#header-spacer {
  width: 100%;
  visibility: hidden;
}
@media print {
  #header-spacer {
    display: none;
  }
}
#ipython_notebook {
  padding-left: 0px;
  padding-top: 1px;
  padding-bottom: 1px;
}
@media (max-width: 991px) {
  #ipython_notebook {
    margin-left: 10px;
  }
}
#noscript {
  width: auto;
  padding-top: 16px;
  padding-bottom: 16px;
  text-align: center;
  font-size: 22px;
  color: red;
  font-weight: bold;
}
#ipython_notebook img {
  height: 28px;
}
#site {
  width: 100%;
  display: none;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  overflow: auto;
}
@media print {
  #site {
    height: auto !important;
  }
}
/* Smaller buttons */
.ui-button .ui-button-text {
  padding: 0.2em 0.8em;
  font-size: 77%;
}
input.ui-button {
  padding: 0.3em 0.9em;
}
span#login_widget {
  float: right;
}
span#login_widget > .button,
#logout {
  color: #333;
  background-color: #fff;
  border-color: #ccc;
}
span#login_widget > .button:focus,
#logout:focus,
span#login_widget > .button.focus,
#logout.focus {
  color: #333;
  background-color: #e6e6e6;
  border-color: #8c8c8c;
}
span#login_widget > .button:hover,
#logout:hover {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
span#login_widget > .button:active,
#logout:active,
span#login_widget > .button.active,
#logout.active,
.open > .dropdown-togglespan#login_widget > .button,
.open > .dropdown-toggle#logout {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
span#login_widget > .button:active:hover,
#logout:active:hover,
span#login_widget > .button.active:hover,
#logout.active:hover,
.open > .dropdown-togglespan#login_widget > .button:hover,
.open > .dropdown-toggle#logout:hover,
span#login_widget > .button:active:focus,
#logout:active:focus,
span#login_widget > .button.active:focus,
#logout.active:focus,
.open > .dropdown-togglespan#login_widget > .button:focus,
.open > .dropdown-toggle#logout:focus,
span#login_widget > .button:active.focus,
#logout:active.focus,
span#login_widget > .button.active.focus,
#logout.active.focus,
.open > .dropdown-togglespan#login_widget > .button.focus,
.open > .dropdown-toggle#logout.focus {
  color: #333;
  background-color: #d4d4d4;
  border-color: #8c8c8c;
}
span#login_widget > .button:active,
#logout:active,
span#login_widget > .button.active,
#logout.active,
.open > .dropdown-togglespan#login_widget > .button,
.open > .dropdown-toggle#logout {
  background-image: none;
}
span#login_widget > .button.disabled:hover,
#logout.disabled:hover,
span#login_widget > .button[disabled]:hover,
#logout[disabled]:hover,
fieldset[disabled] span#login_widget > .button:hover,
fieldset[disabled] #logout:hover,
span#login_widget > .button.disabled:focus,
#logout.disabled:focus,
span#login_widget > .button[disabled]:focus,
#logout[disabled]:focus,
fieldset[disabled] span#login_widget > .button:focus,
fieldset[disabled] #logout:focus,
span#login_widget > .button.disabled.focus,
#logout.disabled.focus,
span#login_widget > .button[disabled].focus,
#logout[disabled].focus,
fieldset[disabled] span#login_widget > .button.focus,
fieldset[disabled] #logout.focus {
  background-color: #fff;
  border-color: #ccc;
}
span#login_widget > .button .badge,
#logout .badge {
  color: #fff;
  background-color: #333;
}
.nav-header {
  text-transform: none;
}
#header > span {
  margin-top: 10px;
}
.modal_stretch .modal-dialog {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  min-height: 80vh;
}
.modal_stretch .modal-dialog .modal-body {
  max-height: calc(100vh - 200px);
  overflow: auto;
  flex: 1;
}
@media (min-width: 768px) {
  .modal .modal-dialog {
    width: 700px;
  }
}
@media (min-width: 768px) {
  select.form-control {
    margin-left: 12px;
    margin-right: 12px;
  }
}
/*!
*
* IPython auth
*
*/
.center-nav {
  display: inline-block;
  margin-bottom: -4px;
}
/*!
*
* IPython tree view
*
*/
/* We need an invisible input field on top of the sentense*/
/* "Drag file onto the list ..." */
.alternate_upload {
  background-color: none;
  display: inline;
}
.alternate_upload.form {
  padding: 0;
  margin: 0;
}
.alternate_upload input.fileinput {
  text-align: center;
  vertical-align: middle;
  display: inline;
  opacity: 0;
  z-index: 2;
  width: 12ex;
  margin-right: -12ex;
}
.alternate_upload .btn-upload {
  height: 22px;
}
/**
 * Primary styles
 *
 * Author: Jupyter Development Team
 */
ul#tabs {
  margin-bottom: 4px;
}
ul#tabs a {
  padding-top: 6px;
  padding-bottom: 4px;
}
ul.breadcrumb a:focus,
ul.breadcrumb a:hover {
  text-decoration: none;
}
ul.breadcrumb i.icon-home {
  font-size: 16px;
  margin-right: 4px;
}
ul.breadcrumb span {
  color: #5e5e5e;
}
.list_toolbar {
  padding: 4px 0 4px 0;
  vertical-align: middle;
}
.list_toolbar .tree-buttons {
  padding-top: 1px;
}
.dynamic-buttons {
  padding-top: 3px;
  display: inline-block;
}
.list_toolbar [class*="span"] {
  min-height: 24px;
}
.list_header {
  font-weight: bold;
  background-color: #EEE;
}
.list_placeholder {
  font-weight: bold;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 7px;
  padding-right: 7px;
}
.list_container {
  margin-top: 4px;
  margin-bottom: 20px;
  border: 1px solid #ddd;
  border-radius: 2px;
}
.list_container > div {
  border-bottom: 1px solid #ddd;
}
.list_container > div:hover .list-item {
  background-color: red;
}
.list_container > div:last-child {
  border: none;
}
.list_item:hover .list_item {
  background-color: #ddd;
}
.list_item a {
  text-decoration: none;
}
.list_item:hover {
  background-color: #fafafa;
}
.list_header > div,
.list_item > div {
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 7px;
  padding-right: 7px;
  line-height: 22px;
}
.list_header > div input,
.list_item > div input {
  margin-right: 7px;
  margin-left: 14px;
  vertical-align: baseline;
  line-height: 22px;
  position: relative;
  top: -1px;
}
.list_header > div .item_link,
.list_item > div .item_link {
  margin-left: -1px;
  vertical-align: baseline;
  line-height: 22px;
}
.new-file input[type=checkbox] {
  visibility: hidden;
}
.item_name {
  line-height: 22px;
  height: 24px;
}
.item_icon {
  font-size: 14px;
  color: #5e5e5e;
  margin-right: 7px;
  margin-left: 7px;
  line-height: 22px;
  vertical-align: baseline;
}
.item_buttons {
  line-height: 1em;
  margin-left: -5px;
}
.item_buttons .btn,
.item_buttons .btn-group,
.item_buttons .input-group {
  float: left;
}
.item_buttons > .btn,
.item_buttons > .btn-group,
.item_buttons > .input-group {
  margin-left: 5px;
}
.item_buttons .btn {
  min-width: 13ex;
}
.item_buttons .running-indicator {
  padding-top: 4px;
  color: #5cb85c;
}
.item_buttons .kernel-name {
  padding-top: 4px;
  color: #5bc0de;
  margin-right: 7px;
  float: left;
}
.toolbar_info {
  height: 24px;
  line-height: 24px;
}
.list_item input:not([type=checkbox]) {
  padding-top: 3px;
  padding-bottom: 3px;
  height: 22px;
  line-height: 14px;
  margin: 0px;
}
.highlight_text {
  color: blue;
}
#project_name {
  display: inline-block;
  padding-left: 7px;
  margin-left: -2px;
}
#project_name > .breadcrumb {
  padding: 0px;
  margin-bottom: 0px;
  background-color: transparent;
  font-weight: bold;
}
#tree-selector {
  padding-right: 0px;
}
#button-select-all {
  min-width: 50px;
}
#select-all {
  margin-left: 7px;
  margin-right: 2px;
}
.menu_icon {
  margin-right: 2px;
}
.tab-content .row {
  margin-left: 0px;
  margin-right: 0px;
}
.folder_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f114";
}
.folder_icon:before.pull-left {
  margin-right: .3em;
}
.folder_icon:before.pull-right {
  margin-left: .3em;
}
.notebook_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f02d";
  position: relative;
  top: -1px;
}
.notebook_icon:before.pull-left {
  margin-right: .3em;
}
.notebook_icon:before.pull-right {
  margin-left: .3em;
}
.running_notebook_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f02d";
  position: relative;
  top: -1px;
  color: #5cb85c;
}
.running_notebook_icon:before.pull-left {
  margin-right: .3em;
}
.running_notebook_icon:before.pull-right {
  margin-left: .3em;
}
.file_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f016";
  position: relative;
  top: -2px;
}
.file_icon:before.pull-left {
  margin-right: .3em;
}
.file_icon:before.pull-right {
  margin-left: .3em;
}
#notebook_toolbar .pull-right {
  padding-top: 0px;
  margin-right: -1px;
}
ul#new-menu {
  left: auto;
  right: 0;
}
.kernel-menu-icon {
  padding-right: 12px;
  width: 24px;
  content: "\f096";
}
.kernel-menu-icon:before {
  content: "\f096";
}
.kernel-menu-icon-current:before {
  content: "\f00c";
}
#tab_content {
  padding-top: 20px;
}
#running .panel-group .panel {
  margin-top: 3px;
  margin-bottom: 1em;
}
#running .panel-group .panel .panel-heading {
  background-color: #EEE;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 7px;
  padding-right: 7px;
  line-height: 22px;
}
#running .panel-group .panel .panel-heading a:focus,
#running .panel-group .panel .panel-heading a:hover {
  text-decoration: none;
}
#running .panel-group .panel .panel-body {
  padding: 0px;
}
#running .panel-group .panel .panel-body .list_container {
  margin-top: 0px;
  margin-bottom: 0px;
  border: 0px;
  border-radius: 0px;
}
#running .panel-group .panel .panel-body .list_container .list_item {
  border-bottom: 1px solid #ddd;
}
#running .panel-group .panel .panel-body .list_container .list_item:last-child {
  border-bottom: 0px;
}
.delete-button {
  display: none;
}
.duplicate-button {
  display: none;
}
.rename-button {
  display: none;
}
.shutdown-button {
  display: none;
}
.dynamic-instructions {
  display: inline-block;
  padding-top: 4px;
}
/*!
*
* IPython text editor webapp
*
*/
.selected-keymap i.fa {
  padding: 0px 5px;
}
.selected-keymap i.fa:before {
  content: "\f00c";
}
#mode-menu {
  overflow: auto;
  max-height: 20em;
}
.edit_app #header {
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
}
.edit_app #menubar .navbar {
  /* Use a negative 1 bottom margin, so the border overlaps the border of the
    header */
  margin-bottom: -1px;
}
.dirty-indicator {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 20px;
}
.dirty-indicator.pull-left {
  margin-right: .3em;
}
.dirty-indicator.pull-right {
  margin-left: .3em;
}
.dirty-indicator-dirty {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 20px;
}
.dirty-indicator-dirty.pull-left {
  margin-right: .3em;
}
.dirty-indicator-dirty.pull-right {
  margin-left: .3em;
}
.dirty-indicator-clean {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 20px;
}
.dirty-indicator-clean.pull-left {
  margin-right: .3em;
}
.dirty-indicator-clean.pull-right {
  margin-left: .3em;
}
.dirty-indicator-clean:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f00c";
}
.dirty-indicator-clean:before.pull-left {
  margin-right: .3em;
}
.dirty-indicator-clean:before.pull-right {
  margin-left: .3em;
}
#filename {
  font-size: 16pt;
  display: table;
  padding: 0px 5px;
}
#current-mode {
  padding-left: 5px;
  padding-right: 5px;
}
#texteditor-backdrop {
  padding-top: 20px;
  padding-bottom: 20px;
}
@media not print {
  #texteditor-backdrop {
    background-color: #EEE;
  }
}
@media print {
  #texteditor-backdrop #texteditor-container .CodeMirror-gutter,
  #texteditor-backdrop #texteditor-container .CodeMirror-gutters {
    background-color: #fff;
  }
}
@media not print {
  #texteditor-backdrop #texteditor-container .CodeMirror-gutter,
  #texteditor-backdrop #texteditor-container .CodeMirror-gutters {
    background-color: #fff;
  }
}
@media not print {
  #texteditor-backdrop #texteditor-container {
    padding: 0px;
    background-color: #fff;
    -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
    box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  }
}
/*!
*
* IPython notebook
*
*/
/* CSS font colors for translated ANSI colors. */
.ansibold {
  font-weight: bold;
}
/* use dark versions for foreground, to improve visibility */
.ansiblack {
  color: black;
}
.ansired {
  color: darkred;
}
.ansigreen {
  color: darkgreen;
}
.ansiyellow {
  color: #c4a000;
}
.ansiblue {
  color: darkblue;
}
.ansipurple {
  color: darkviolet;
}
.ansicyan {
  color: steelblue;
}
.ansigray {
  color: gray;
}
/* and light for background, for the same reason */
.ansibgblack {
  background-color: black;
}
.ansibgred {
  background-color: red;
}
.ansibggreen {
  background-color: green;
}
.ansibgyellow {
  background-color: yellow;
}
.ansibgblue {
  background-color: blue;
}
.ansibgpurple {
  background-color: magenta;
}
.ansibgcyan {
  background-color: cyan;
}
.ansibggray {
  background-color: gray;
}
div.cell {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  border-radius: 2px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  border-width: 1px;
  border-style: solid;
  border-color: transparent;
  width: 100%;
  padding: 5px;
  /* This acts as a spacer between cells, that is outside the border */
  margin: 0px;
  outline: none;
  border-left-width: 1px;
  padding-left: 5px;
  background: linear-gradient(to right, transparent -40px, transparent 1px, transparent 1px, transparent 100%);
}
div.cell.jupyter-soft-selected {
  border-left-color: #90CAF9;
  border-left-color: #E3F2FD;
  border-left-width: 1px;
  padding-left: 5px;
  border-right-color: #E3F2FD;
  border-right-width: 1px;
  background: #E3F2FD;
}
@media print {
  div.cell.jupyter-soft-selected {
    border-color: transparent;
  }
}
div.cell.selected {
  border-color: #ababab;
  border-left-width: 0px;
  padding-left: 6px;
  background: linear-gradient(to right, #42A5F5 -40px, #42A5F5 5px, transparent 5px, transparent 100%);
}
@media print {
  div.cell.selected {
    border-color: transparent;
  }
}
div.cell.selected.jupyter-soft-selected {
  border-left-width: 0;
  padding-left: 6px;
  background: linear-gradient(to right, #42A5F5 -40px, #42A5F5 7px, #E3F2FD 7px, #E3F2FD 100%);
}
.edit_mode div.cell.selected {
  border-color: #66BB6A;
  border-left-width: 0px;
  padding-left: 6px;
  background: linear-gradient(to right, #66BB6A -40px, #66BB6A 5px, transparent 5px, transparent 100%);
}
@media print {
  .edit_mode div.cell.selected {
    border-color: transparent;
  }
}
.prompt {
  /* This needs to be wide enough for 3 digit prompt numbers: In[100]: */
  min-width: 14ex;
  /* This padding is tuned to match the padding on the CodeMirror editor. */
  padding: 0.4em;
  margin: 0px;
  font-family: monospace;
  text-align: right;
  /* This has to match that of the the CodeMirror class line-height below */
  line-height: 1.21429em;
  /* Don't highlight prompt number selection */
  -webkit-touch-callout: none;
  -webkit-user-select: none;
  -khtml-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
  /* Use default cursor */
  cursor: default;
}
@media (max-width: 540px) {
  .prompt {
    text-align: left;
  }
}
div.inner_cell {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
@-moz-document url-prefix() {
  div.inner_cell {
    overflow-x: hidden;
  }
}
/* input_area and input_prompt must match in top border and margin for alignment */
div.input_area {
  border: 1px solid #cfcfcf;
  border-radius: 2px;
  background: #f7f7f7;
  line-height: 1.21429em;
}
/* This is needed so that empty prompt areas can collapse to zero height when there
   is no content in the output_subarea and the prompt. The main purpose of this is
   to make sure that empty JavaScript output_subareas have no height. */
div.prompt:empty {
  padding-top: 0;
  padding-bottom: 0;
}
div.unrecognized_cell {
  padding: 5px 5px 5px 0px;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
div.unrecognized_cell .inner_cell {
  border-radius: 2px;
  padding: 5px;
  font-weight: bold;
  color: red;
  border: 1px solid #cfcfcf;
  background: #eaeaea;
}
div.unrecognized_cell .inner_cell a {
  color: inherit;
  text-decoration: none;
}
div.unrecognized_cell .inner_cell a:hover {
  color: inherit;
  text-decoration: none;
}
@media (max-width: 540px) {
  div.unrecognized_cell > div.prompt {
    display: none;
  }
}
div.code_cell {
  /* avoid page breaking on code cells when printing */
}
@media print {
  div.code_cell {
    page-break-inside: avoid;
  }
}
/* any special styling for code cells that are currently running goes here */
div.input {
  page-break-inside: avoid;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.input {
    /* Old browsers */
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-box-align: stretch;
    display: -moz-box;
    -moz-box-orient: vertical;
    -moz-box-align: stretch;
    display: box;
    box-orient: vertical;
    box-align: stretch;
    /* Modern browsers */
    display: flex;
    flex-direction: column;
    align-items: stretch;
  }
}
/* input_area and input_prompt must match in top border and margin for alignment */
div.input_prompt {
  color: #303F9F;
  border-top: 1px solid transparent;
}
div.input_area > div.highlight {
  margin: 0.4em;
  border: none;
  padding: 0px;
  background-color: transparent;
}
div.input_area > div.highlight > pre {
  margin: 0px;
  border: none;
  padding: 0px;
  background-color: transparent;
}
/* The following gets added to the <head> if it is detected that the user has a
 * monospace font with inconsistent normal/bold/italic height.  See
 * notebookmain.js.  Such fonts will have keywords vertically offset with
 * respect to the rest of the text.  The user should select a better font.
 * See: https://github.com/ipython/ipython/issues/1503
 *
 * .CodeMirror span {
 *      vertical-align: bottom;
 * }
 */
.CodeMirror {
  line-height: 1.21429em;
  /* Changed from 1em to our global default */
  font-size: 14px;
  height: auto;
  /* Changed to auto to autogrow */
  background: none;
  /* Changed from white to allow our bg to show through */
}
.CodeMirror-scroll {
  /*  The CodeMirror docs are a bit fuzzy on if overflow-y should be hidden or visible.*/
  /*  We have found that if it is visible, vertical scrollbars appear with font size changes.*/
  overflow-y: hidden;
  overflow-x: auto;
}
.CodeMirror-lines {
  /* In CM2, this used to be 0.4em, but in CM3 it went to 4px. We need the em value because */
  /* we have set a different line-height and want this to scale with that. */
  padding: 0.4em;
}
.CodeMirror-linenumber {
  padding: 0 8px 0 4px;
}
.CodeMirror-gutters {
  border-bottom-left-radius: 2px;
  border-top-left-radius: 2px;
}
.CodeMirror pre {
  /* In CM3 this went to 4px from 0 in CM2. We need the 0 value because of how we size */
  /* .CodeMirror-lines */
  padding: 0;
  border: 0;
  border-radius: 0;
}
/*

Original style from softwaremaniacs.org (c) Ivan Sagalaev <Maniac@SoftwareManiacs.Org>
Adapted from GitHub theme

*/
.highlight-base {
  color: #000;
}
.highlight-variable {
  color: #000;
}
.highlight-variable-2 {
  color: #1a1a1a;
}
.highlight-variable-3 {
  color: #333333;
}
.highlight-string {
  color: #BA2121;
}
.highlight-comment {
  color: #408080;
  font-style: italic;
}
.highlight-number {
  color: #080;
}
.highlight-atom {
  color: #88F;
}
.highlight-keyword {
  color: #008000;
  font-weight: bold;
}
.highlight-builtin {
  color: #008000;
}
.highlight-error {
  color: #f00;
}
.highlight-operator {
  color: #AA22FF;
  font-weight: bold;
}
.highlight-meta {
  color: #AA22FF;
}
/* previously not defined, copying from default codemirror */
.highlight-def {
  color: #00f;
}
.highlight-string-2 {
  color: #f50;
}
.highlight-qualifier {
  color: #555;
}
.highlight-bracket {
  color: #997;
}
.highlight-tag {
  color: #170;
}
.highlight-attribute {
  color: #00c;
}
.highlight-header {
  color: blue;
}
.highlight-quote {
  color: #090;
}
.highlight-link {
  color: #00c;
}
/* apply the same style to codemirror */
.cm-s-ipython span.cm-keyword {
  color: #008000;
  font-weight: bold;
}
.cm-s-ipython span.cm-atom {
  color: #88F;
}
.cm-s-ipython span.cm-number {
  color: #080;
}
.cm-s-ipython span.cm-def {
  color: #00f;
}
.cm-s-ipython span.cm-variable {
  color: #000;
}
.cm-s-ipython span.cm-operator {
  color: #AA22FF;
  font-weight: bold;
}
.cm-s-ipython span.cm-variable-2 {
  color: #1a1a1a;
}
.cm-s-ipython span.cm-variable-3 {
  color: #333333;
}
.cm-s-ipython span.cm-comment {
  color: #408080;
  font-style: italic;
}
.cm-s-ipython span.cm-string {
  color: #BA2121;
}
.cm-s-ipython span.cm-string-2 {
  color: #f50;
}
.cm-s-ipython span.cm-meta {
  color: #AA22FF;
}
.cm-s-ipython span.cm-qualifier {
  color: #555;
}
.cm-s-ipython span.cm-builtin {
  color: #008000;
}
.cm-s-ipython span.cm-bracket {
  color: #997;
}
.cm-s-ipython span.cm-tag {
  color: #170;
}
.cm-s-ipython span.cm-attribute {
  color: #00c;
}
.cm-s-ipython span.cm-header {
  color: blue;
}
.cm-s-ipython span.cm-quote {
  color: #090;
}
.cm-s-ipython span.cm-link {
  color: #00c;
}
.cm-s-ipython span.cm-error {
  color: #f00;
}
.cm-s-ipython span.cm-tab {
  background: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAMCAYAAAAkuj5RAAAAAXNSR0IArs4c6QAAAGFJREFUSMft1LsRQFAQheHPowAKoACx3IgEKtaEHujDjORSgWTH/ZOdnZOcM/sgk/kFFWY0qV8foQwS4MKBCS3qR6ixBJvElOobYAtivseIE120FaowJPN75GMu8j/LfMwNjh4HUpwg4LUAAAAASUVORK5CYII=);
  background-position: right;
  background-repeat: no-repeat;
}
div.output_wrapper {
  /* this position must be relative to enable descendents to be absolute within it */
  position: relative;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  z-index: 1;
}
/* class for the output area when it should be height-limited */
div.output_scroll {
  /* ideally, this would be max-height, but FF barfs all over that */
  height: 24em;
  /* FF needs this *and the wrapper* to specify full width, or it will shrinkwrap */
  width: 100%;
  overflow: auto;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.8);
  box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.8);
  display: block;
}
/* output div while it is collapsed */
div.output_collapsed {
  margin: 0px;
  padding: 0px;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
div.out_prompt_overlay {
  height: 100%;
  padding: 0px 0.4em;
  position: absolute;
  border-radius: 2px;
}
div.out_prompt_overlay:hover {
  /* use inner shadow to get border that is computed the same on WebKit/FF */
  -webkit-box-shadow: inset 0 0 1px #000;
  box-shadow: inset 0 0 1px #000;
  background: rgba(240, 240, 240, 0.5);
}
div.output_prompt {
  color: #D84315;
}
/* This class is the outer container of all output sections. */
div.output_area {
  padding: 0px;
  page-break-inside: avoid;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
div.output_area .MathJax_Display {
  text-align: left !important;
}
div.output_area .rendered_html table {
  margin-left: 0;
  margin-right: 0;
}
div.output_area .rendered_html img {
  margin-left: 0;
  margin-right: 0;
}
div.output_area img,
div.output_area svg {
  max-width: 100%;
  height: auto;
}
div.output_area img.unconfined,
div.output_area svg.unconfined {
  max-width: none;
}
/* This is needed to protect the pre formating from global settings such
   as that of bootstrap */
.output {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.output_area {
    /* Old browsers */
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-box-align: stretch;
    display: -moz-box;
    -moz-box-orient: vertical;
    -moz-box-align: stretch;
    display: box;
    box-orient: vertical;
    box-align: stretch;
    /* Modern browsers */
    display: flex;
    flex-direction: column;
    align-items: stretch;
  }
}
div.output_area pre {
  margin: 0;
  padding: 0;
  border: 0;
  vertical-align: baseline;
  color: black;
  background-color: transparent;
  border-radius: 0;
}
/* This class is for the output subarea inside the output_area and after
   the prompt div. */
div.output_subarea {
  overflow-x: auto;
  padding: 0.4em;
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
  max-width: calc(100% - 14ex);
}
div.output_scroll div.output_subarea {
  overflow-x: visible;
}
/* The rest of the output_* classes are for special styling of the different
   output types */
/* all text output has this class: */
div.output_text {
  text-align: left;
  color: #000;
  /* This has to match that of the the CodeMirror class line-height below */
  line-height: 1.21429em;
}
/* stdout/stderr are 'text' as well as 'stream', but execute_result/error are *not* streams */
div.output_stderr {
  background: #fdd;
  /* very light red background for stderr */
}
div.output_latex {
  text-align: left;
}
/* Empty output_javascript divs should have no height */
div.output_javascript:empty {
  padding: 0;
}
.js-error {
  color: darkred;
}
/* raw_input styles */
div.raw_input_container {
  line-height: 1.21429em;
  padding-top: 5px;
}
pre.raw_input_prompt {
  /* nothing needed here. */
}
input.raw_input {
  font-family: monospace;
  font-size: inherit;
  color: inherit;
  width: auto;
  /* make sure input baseline aligns with prompt */
  vertical-align: baseline;
  /* padding + margin = 0.5em between prompt and cursor */
  padding: 0em 0.25em;
  margin: 0em 0.25em;
}
input.raw_input:focus {
  box-shadow: none;
}
p.p-space {
  margin-bottom: 10px;
}
div.output_unrecognized {
  padding: 5px;
  font-weight: bold;
  color: red;
}
div.output_unrecognized a {
  color: inherit;
  text-decoration: none;
}
div.output_unrecognized a:hover {
  color: inherit;
  text-decoration: none;
}
.rendered_html {
  color: #000;
  /* any extras will just be numbers: */
}
.rendered_html em {
  font-style: italic;
}
.rendered_html strong {
  font-weight: bold;
}
.rendered_html u {
  text-decoration: underline;
}
.rendered_html :link {
  text-decoration: underline;
}
.rendered_html :visited {
  text-decoration: underline;
}
.rendered_html h1 {
  font-size: 185.7%;
  margin: 1.08em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h2 {
  font-size: 157.1%;
  margin: 1.27em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h3 {
  font-size: 128.6%;
  margin: 1.55em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h4 {
  font-size: 100%;
  margin: 2em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h5 {
  font-size: 100%;
  margin: 2em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
  font-style: italic;
}
.rendered_html h6 {
  font-size: 100%;
  margin: 2em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
  font-style: italic;
}
.rendered_html h1:first-child {
  margin-top: 0.538em;
}
.rendered_html h2:first-child {
  margin-top: 0.636em;
}
.rendered_html h3:first-child {
  margin-top: 0.777em;
}
.rendered_html h4:first-child {
  margin-top: 1em;
}
.rendered_html h5:first-child {
  margin-top: 1em;
}
.rendered_html h6:first-child {
  margin-top: 1em;
}
.rendered_html ul {
  list-style: disc;
  margin: 0em 2em;
  padding-left: 0px;
}
.rendered_html ul ul {
  list-style: square;
  margin: 0em 2em;
}
.rendered_html ul ul ul {
  list-style: circle;
  margin: 0em 2em;
}
.rendered_html ol {
  list-style: decimal;
  margin: 0em 2em;
  padding-left: 0px;
}
.rendered_html ol ol {
  list-style: upper-alpha;
  margin: 0em 2em;
}
.rendered_html ol ol ol {
  list-style: lower-alpha;
  margin: 0em 2em;
}
.rendered_html ol ol ol ol {
  list-style: lower-roman;
  margin: 0em 2em;
}
.rendered_html ol ol ol ol ol {
  list-style: decimal;
  margin: 0em 2em;
}
.rendered_html * + ul {
  margin-top: 1em;
}
.rendered_html * + ol {
  margin-top: 1em;
}
.rendered_html hr {
  color: black;
  background-color: black;
}
.rendered_html pre {
  margin: 1em 2em;
}
.rendered_html pre,
.rendered_html code {
  border: 0;
  background-color: #fff;
  color: #000;
  font-size: 100%;
  padding: 0px;
}
.rendered_html blockquote {
  margin: 1em 2em;
}
.rendered_html table {
  margin-left: auto;
  margin-right: auto;
  border: 1px solid black;
  border-collapse: collapse;
}
.rendered_html tr,
.rendered_html th,
.rendered_html td {
  border: 1px solid black;
  border-collapse: collapse;
  margin: 1em 2em;
}
.rendered_html td,
.rendered_html th {
  text-align: left;
  vertical-align: middle;
  padding: 4px;
}
.rendered_html th {
  font-weight: bold;
}
.rendered_html * + table {
  margin-top: 1em;
}
.rendered_html p {
  text-align: left;
}
.rendered_html * + p {
  margin-top: 1em;
}
.rendered_html img {
  display: block;
  margin-left: auto;
  margin-right: auto;
}
.rendered_html * + img {
  margin-top: 1em;
}
.rendered_html img,
.rendered_html svg {
  max-width: 100%;
  height: auto;
}
.rendered_html img.unconfined,
.rendered_html svg.unconfined {
  max-width: none;
}
div.text_cell {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.text_cell > div.prompt {
    display: none;
  }
}
div.text_cell_render {
  /*font-family: "Helvetica Neue", Arial, Helvetica, Geneva, sans-serif;*/
  outline: none;
  resize: none;
  width: inherit;
  border-style: none;
  padding: 0.5em 0.5em 0.5em 0.4em;
  color: #000;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
a.anchor-link:link {
  text-decoration: none;
  padding: 0px 20px;
  visibility: hidden;
}
h1:hover .anchor-link,
h2:hover .anchor-link,
h3:hover .anchor-link,
h4:hover .anchor-link,
h5:hover .anchor-link,
h6:hover .anchor-link {
  visibility: visible;
}
.text_cell.rendered .input_area {
  display: none;
}
.text_cell.rendered .rendered_html {
  overflow-x: auto;
  overflow-y: hidden;
}
.text_cell.unrendered .text_cell_render {
  display: none;
}
.cm-header-1,
.cm-header-2,
.cm-header-3,
.cm-header-4,
.cm-header-5,
.cm-header-6 {
  font-weight: bold;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
}
.cm-header-1 {
  font-size: 185.7%;
}
.cm-header-2 {
  font-size: 157.1%;
}
.cm-header-3 {
  font-size: 128.6%;
}
.cm-header-4 {
  font-size: 110%;
}
.cm-header-5 {
  font-size: 100%;
  font-style: italic;
}
.cm-header-6 {
  font-size: 100%;
  font-style: italic;
}
/*!
*
* IPython notebook webapp
*
*/
@media (max-width: 767px) {
  .notebook_app {
    padding-left: 0px;
    padding-right: 0px;
  }
}
#ipython-main-app {
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  height: 100%;
}
div#notebook_panel {
  margin: 0px;
  padding: 0px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  height: 100%;
}
div#notebook {
  font-size: 14px;
  line-height: 20px;
  overflow-y: hidden;
  overflow-x: auto;
  width: 100%;
  /* This spaces the page away from the edge of the notebook area */
  padding-top: 20px;
  margin: 0px;
  outline: none;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  min-height: 100%;
}
@media not print {
  #notebook-container {
    padding: 15px;
    background-color: #fff;
    min-height: 0;
    -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
    box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  }
}
@media print {
  #notebook-container {
    width: 100%;
  }
}
div.ui-widget-content {
  border: 1px solid #ababab;
  outline: none;
}
pre.dialog {
  background-color: #f7f7f7;
  border: 1px solid #ddd;
  border-radius: 2px;
  padding: 0.4em;
  padding-left: 2em;
}
p.dialog {
  padding: 0.2em;
}
/* Word-wrap output correctly.  This is the CSS3 spelling, though Firefox seems
   to not honor it correctly.  Webkit browsers (Chrome, rekonq, Safari) do.
 */
pre,
code,
kbd,
samp {
  white-space: pre-wrap;
}
#fonttest {
  font-family: monospace;
}
p {
  margin-bottom: 0;
}
.end_space {
  min-height: 100px;
  transition: height .2s ease;
}
.notebook_app > #header {
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
}
@media not print {
  .notebook_app {
    background-color: #EEE;
  }
}
kbd {
  border-style: solid;
  border-width: 1px;
  box-shadow: none;
  margin: 2px;
  padding-left: 2px;
  padding-right: 2px;
  padding-top: 1px;
  padding-bottom: 1px;
}
/* CSS for the cell toolbar */
.celltoolbar {
  border: thin solid #CFCFCF;
  border-bottom: none;
  background: #EEE;
  border-radius: 2px 2px 0px 0px;
  width: 100%;
  height: 29px;
  padding-right: 4px;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
  /* Old browsers */
  -webkit-box-pack: end;
  -moz-box-pack: end;
  box-pack: end;
  /* Modern browsers */
  justify-content: flex-end;
  display: -webkit-flex;
}
@media print {
  .celltoolbar {
    display: none;
  }
}
.ctb_hideshow {
  display: none;
  vertical-align: bottom;
}
/* ctb_show is added to the ctb_hideshow div to show the cell toolbar.
   Cell toolbars are only shown when the ctb_global_show class is also set.
*/
.ctb_global_show .ctb_show.ctb_hideshow {
  display: block;
}
.ctb_global_show .ctb_show + .input_area,
.ctb_global_show .ctb_show + div.text_cell_input,
.ctb_global_show .ctb_show ~ div.text_cell_render {
  border-top-right-radius: 0px;
  border-top-left-radius: 0px;
}
.ctb_global_show .ctb_show ~ div.text_cell_render {
  border: 1px solid #cfcfcf;
}
.celltoolbar {
  font-size: 87%;
  padding-top: 3px;
}
.celltoolbar select {
  display: block;
  width: 100%;
  height: 32px;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
  background-color: #fff;
  background-image: none;
  border: 1px solid #ccc;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  -webkit-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  -o-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
  width: inherit;
  font-size: inherit;
  height: 22px;
  padding: 0px;
  display: inline-block;
}
.celltoolbar select:focus {
  border-color: #66afe9;
  outline: 0;
  -webkit-box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
  box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
}
.celltoolbar select::-moz-placeholder {
  color: #999;
  opacity: 1;
}
.celltoolbar select:-ms-input-placeholder {
  color: #999;
}
.celltoolbar select::-webkit-input-placeholder {
  color: #999;
}
.celltoolbar select::-ms-expand {
  border: 0;
  background-color: transparent;
}
.celltoolbar select[disabled],
.celltoolbar select[readonly],
fieldset[disabled] .celltoolbar select {
  background-color: #eeeeee;
  opacity: 1;
}
.celltoolbar select[disabled],
fieldset[disabled] .celltoolbar select {
  cursor: not-allowed;
}
textarea.celltoolbar select {
  height: auto;
}
select.celltoolbar select {
  height: 30px;
  line-height: 30px;
}
textarea.celltoolbar select,
select[multiple].celltoolbar select {
  height: auto;
}
.celltoolbar label {
  margin-left: 5px;
  margin-right: 5px;
}
.completions {
  position: absolute;
  z-index: 110;
  overflow: hidden;
  border: 1px solid #ababab;
  border-radius: 2px;
  -webkit-box-shadow: 0px 6px 10px -1px #adadad;
  box-shadow: 0px 6px 10px -1px #adadad;
  line-height: 1;
}
.completions select {
  background: white;
  outline: none;
  border: none;
  padding: 0px;
  margin: 0px;
  overflow: auto;
  font-family: monospace;
  font-size: 110%;
  color: #000;
  width: auto;
}
.completions select option.context {
  color: #286090;
}
#kernel_logo_widget {
  float: right !important;
  float: right;
}
#kernel_logo_widget .current_kernel_logo {
  display: none;
  margin-top: -1px;
  margin-bottom: -1px;
  width: 32px;
  height: 32px;
}
#menubar {
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  margin-top: 1px;
}
#menubar .navbar {
  border-top: 1px;
  border-radius: 0px 0px 2px 2px;
  margin-bottom: 0px;
}
#menubar .navbar-toggle {
  float: left;
  padding-top: 7px;
  padding-bottom: 7px;
  border: none;
}
#menubar .navbar-collapse {
  clear: left;
}
.nav-wrapper {
  border-bottom: 1px solid #e7e7e7;
}
i.menu-icon {
  padding-top: 4px;
}
ul#help_menu li a {
  overflow: hidden;
  padding-right: 2.2em;
}
ul#help_menu li a i {
  margin-right: -1.2em;
}
.dropdown-submenu {
  position: relative;
}
.dropdown-submenu > .dropdown-menu {
  top: 0;
  left: 100%;
  margin-top: -6px;
  margin-left: -1px;
}
.dropdown-submenu:hover > .dropdown-menu {
  display: block;
}
.dropdown-submenu > a:after {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  display: block;
  content: "\f0da";
  float: right;
  color: #333333;
  margin-top: 2px;
  margin-right: -10px;
}
.dropdown-submenu > a:after.pull-left {
  margin-right: .3em;
}
.dropdown-submenu > a:after.pull-right {
  margin-left: .3em;
}
.dropdown-submenu:hover > a:after {
  color: #262626;
}
.dropdown-submenu.pull-left {
  float: none;
}
.dropdown-submenu.pull-left > .dropdown-menu {
  left: -100%;
  margin-left: 10px;
}
#notification_area {
  float: right !important;
  float: right;
  z-index: 10;
}
.indicator_area {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
}
#kernel_indicator {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
  border-left: 1px solid;
}
#kernel_indicator .kernel_indicator_name {
  padding-left: 5px;
  padding-right: 5px;
}
#modal_indicator {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
}
#readonly-indicator {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
  margin-top: 2px;
  margin-bottom: 0px;
  margin-left: 0px;
  margin-right: 0px;
  display: none;
}
.modal_indicator:before {
  width: 1.28571429em;
  text-align: center;
}
.edit_mode .modal_indicator:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f040";
}
.edit_mode .modal_indicator:before.pull-left {
  margin-right: .3em;
}
.edit_mode .modal_indicator:before.pull-right {
  margin-left: .3em;
}
.command_mode .modal_indicator:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: ' ';
}
.command_mode .modal_indicator:before.pull-left {
  margin-right: .3em;
}
.command_mode .modal_indicator:before.pull-right {
  margin-left: .3em;
}
.kernel_idle_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f10c";
}
.kernel_idle_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_idle_icon:before.pull-right {
  margin-left: .3em;
}
.kernel_busy_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f111";
}
.kernel_busy_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_busy_icon:before.pull-right {
  margin-left: .3em;
}
.kernel_dead_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f1e2";
}
.kernel_dead_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_dead_icon:before.pull-right {
  margin-left: .3em;
}
.kernel_disconnected_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f127";
}
.kernel_disconnected_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_disconnected_icon:before.pull-right {
  margin-left: .3em;
}
.notification_widget {
  color: #777;
  z-index: 10;
  background: rgba(240, 240, 240, 0.5);
  margin-right: 4px;
  color: #333;
  background-color: #fff;
  border-color: #ccc;
}
.notification_widget:focus,
.notification_widget.focus {
  color: #333;
  background-color: #e6e6e6;
  border-color: #8c8c8c;
}
.notification_widget:hover {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.notification_widget:active,
.notification_widget.active,
.open > .dropdown-toggle.notification_widget {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.notification_widget:active:hover,
.notification_widget.active:hover,
.open > .dropdown-toggle.notification_widget:hover,
.notification_widget:active:focus,
.notification_widget.active:focus,
.open > .dropdown-toggle.notification_widget:focus,
.notification_widget:active.focus,
.notification_widget.active.focus,
.open > .dropdown-toggle.notification_widget.focus {
  color: #333;
  background-color: #d4d4d4;
  border-color: #8c8c8c;
}
.notification_widget:active,
.notification_widget.active,
.open > .dropdown-toggle.notification_widget {
  background-image: none;
}
.notification_widget.disabled:hover,
.notification_widget[disabled]:hover,
fieldset[disabled] .notification_widget:hover,
.notification_widget.disabled:focus,
.notification_widget[disabled]:focus,
fieldset[disabled] .notification_widget:focus,
.notification_widget.disabled.focus,
.notification_widget[disabled].focus,
fieldset[disabled] .notification_widget.focus {
  background-color: #fff;
  border-color: #ccc;
}
.notification_widget .badge {
  color: #fff;
  background-color: #333;
}
.notification_widget.warning {
  color: #fff;
  background-color: #f0ad4e;
  border-color: #eea236;
}
.notification_widget.warning:focus,
.notification_widget.warning.focus {
  color: #fff;
  background-color: #ec971f;
  border-color: #985f0d;
}
.notification_widget.warning:hover {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.notification_widget.warning:active,
.notification_widget.warning.active,
.open > .dropdown-toggle.notification_widget.warning {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.notification_widget.warning:active:hover,
.notification_widget.warning.active:hover,
.open > .dropdown-toggle.notification_widget.warning:hover,
.notification_widget.warning:active:focus,
.notification_widget.warning.active:focus,
.open > .dropdown-toggle.notification_widget.warning:focus,
.notification_widget.warning:active.focus,
.notification_widget.warning.active.focus,
.open > .dropdown-toggle.notification_widget.warning.focus {
  color: #fff;
  background-color: #d58512;
  border-color: #985f0d;
}
.notification_widget.warning:active,
.notification_widget.warning.active,
.open > .dropdown-toggle.notification_widget.warning {
  background-image: none;
}
.notification_widget.warning.disabled:hover,
.notification_widget.warning[disabled]:hover,
fieldset[disabled] .notification_widget.warning:hover,
.notification_widget.warning.disabled:focus,
.notification_widget.warning[disabled]:focus,
fieldset[disabled] .notification_widget.warning:focus,
.notification_widget.warning.disabled.focus,
.notification_widget.warning[disabled].focus,
fieldset[disabled] .notification_widget.warning.focus {
  background-color: #f0ad4e;
  border-color: #eea236;
}
.notification_widget.warning .badge {
  color: #f0ad4e;
  background-color: #fff;
}
.notification_widget.success {
  color: #fff;
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.notification_widget.success:focus,
.notification_widget.success.focus {
  color: #fff;
  background-color: #449d44;
  border-color: #255625;
}
.notification_widget.success:hover {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.notification_widget.success:active,
.notification_widget.success.active,
.open > .dropdown-toggle.notification_widget.success {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.notification_widget.success:active:hover,
.notification_widget.success.active:hover,
.open > .dropdown-toggle.notification_widget.success:hover,
.notification_widget.success:active:focus,
.notification_widget.success.active:focus,
.open > .dropdown-toggle.notification_widget.success:focus,
.notification_widget.success:active.focus,
.notification_widget.success.active.focus,
.open > .dropdown-toggle.notification_widget.success.focus {
  color: #fff;
  background-color: #398439;
  border-color: #255625;
}
.notification_widget.success:active,
.notification_widget.success.active,
.open > .dropdown-toggle.notification_widget.success {
  background-image: none;
}
.notification_widget.success.disabled:hover,
.notification_widget.success[disabled]:hover,
fieldset[disabled] .notification_widget.success:hover,
.notification_widget.success.disabled:focus,
.notification_widget.success[disabled]:focus,
fieldset[disabled] .notification_widget.success:focus,
.notification_widget.success.disabled.focus,
.notification_widget.success[disabled].focus,
fieldset[disabled] .notification_widget.success.focus {
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.notification_widget.success .badge {
  color: #5cb85c;
  background-color: #fff;
}
.notification_widget.info {
  color: #fff;
  background-color: #5bc0de;
  border-color: #46b8da;
}
.notification_widget.info:focus,
.notification_widget.info.focus {
  color: #fff;
  background-color: #31b0d5;
  border-color: #1b6d85;
}
.notification_widget.info:hover {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.notification_widget.info:active,
.notification_widget.info.active,
.open > .dropdown-toggle.notification_widget.info {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.notification_widget.info:active:hover,
.notification_widget.info.active:hover,
.open > .dropdown-toggle.notification_widget.info:hover,
.notification_widget.info:active:focus,
.notification_widget.info.active:focus,
.open > .dropdown-toggle.notification_widget.info:focus,
.notification_widget.info:active.focus,
.notification_widget.info.active.focus,
.open > .dropdown-toggle.notification_widget.info.focus {
  color: #fff;
  background-color: #269abc;
  border-color: #1b6d85;
}
.notification_widget.info:active,
.notification_widget.info.active,
.open > .dropdown-toggle.notification_widget.info {
  background-image: none;
}
.notification_widget.info.disabled:hover,
.notification_widget.info[disabled]:hover,
fieldset[disabled] .notification_widget.info:hover,
.notification_widget.info.disabled:focus,
.notification_widget.info[disabled]:focus,
fieldset[disabled] .notification_widget.info:focus,
.notification_widget.info.disabled.focus,
.notification_widget.info[disabled].focus,
fieldset[disabled] .notification_widget.info.focus {
  background-color: #5bc0de;
  border-color: #46b8da;
}
.notification_widget.info .badge {
  color: #5bc0de;
  background-color: #fff;
}
.notification_widget.danger {
  color: #fff;
  background-color: #d9534f;
  border-color: #d43f3a;
}
.notification_widget.danger:focus,
.notification_widget.danger.focus {
  color: #fff;
  background-color: #c9302c;
  border-color: #761c19;
}
.notification_widget.danger:hover {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.notification_widget.danger:active,
.notification_widget.danger.active,
.open > .dropdown-toggle.notification_widget.danger {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.notification_widget.danger:active:hover,
.notification_widget.danger.active:hover,
.open > .dropdown-toggle.notification_widget.danger:hover,
.notification_widget.danger:active:focus,
.notification_widget.danger.active:focus,
.open > .dropdown-toggle.notification_widget.danger:focus,
.notification_widget.danger:active.focus,
.notification_widget.danger.active.focus,
.open > .dropdown-toggle.notification_widget.danger.focus {
  color: #fff;
  background-color: #ac2925;
  border-color: #761c19;
}
.notification_widget.danger:active,
.notification_widget.danger.active,
.open > .dropdown-toggle.notification_widget.danger {
  background-image: none;
}
.notification_widget.danger.disabled:hover,
.notification_widget.danger[disabled]:hover,
fieldset[disabled] .notification_widget.danger:hover,
.notification_widget.danger.disabled:focus,
.notification_widget.danger[disabled]:focus,
fieldset[disabled] .notification_widget.danger:focus,
.notification_widget.danger.disabled.focus,
.notification_widget.danger[disabled].focus,
fieldset[disabled] .notification_widget.danger.focus {
  background-color: #d9534f;
  border-color: #d43f3a;
}
.notification_widget.danger .badge {
  color: #d9534f;
  background-color: #fff;
}
div#pager {
  background-color: #fff;
  font-size: 14px;
  line-height: 20px;
  overflow: hidden;
  display: none;
  position: fixed;
  bottom: 0px;
  width: 100%;
  max-height: 50%;
  padding-top: 8px;
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  /* Display over codemirror */
  z-index: 100;
  /* Hack which prevents jquery ui resizable from changing top. */
  top: auto !important;
}
div#pager pre {
  line-height: 1.21429em;
  color: #000;
  background-color: #f7f7f7;
  padding: 0.4em;
}
div#pager #pager-button-area {
  position: absolute;
  top: 8px;
  right: 20px;
}
div#pager #pager-contents {
  position: relative;
  overflow: auto;
  width: 100%;
  height: 100%;
}
div#pager #pager-contents #pager-container {
  position: relative;
  padding: 15px 0px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
div#pager .ui-resizable-handle {
  top: 0px;
  height: 8px;
  background: #f7f7f7;
  border-top: 1px solid #cfcfcf;
  border-bottom: 1px solid #cfcfcf;
  /* This injects handle bars (a short, wide = symbol) for 
        the resize handle. */
}
div#pager .ui-resizable-handle::after {
  content: '';
  top: 2px;
  left: 50%;
  height: 3px;
  width: 30px;
  margin-left: -15px;
  position: absolute;
  border-top: 1px solid #cfcfcf;
}
.quickhelp {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
  line-height: 1.8em;
}
.shortcut_key {
  display: inline-block;
  width: 20ex;
  text-align: right;
  font-family: monospace;
}
.shortcut_descr {
  display: inline-block;
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
span.save_widget {
  margin-top: 6px;
}
span.save_widget span.filename {
  height: 1em;
  line-height: 1em;
  padding: 3px;
  margin-left: 16px;
  border: none;
  font-size: 146.5%;
  border-radius: 2px;
}
span.save_widget span.filename:hover {
  background-color: #e6e6e6;
}
span.checkpoint_status,
span.autosave_status {
  font-size: small;
}
@media (max-width: 767px) {
  span.save_widget {
    font-size: small;
  }
  span.checkpoint_status,
  span.autosave_status {
    display: none;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  span.checkpoint_status {
    display: none;
  }
  span.autosave_status {
    font-size: x-small;
  }
}
.toolbar {
  padding: 0px;
  margin-left: -5px;
  margin-top: 2px;
  margin-bottom: 5px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
.toolbar select,
.toolbar label {
  width: auto;
  vertical-align: middle;
  margin-right: 2px;
  margin-bottom: 0px;
  display: inline;
  font-size: 92%;
  margin-left: 0.3em;
  margin-right: 0.3em;
  padding: 0px;
  padding-top: 3px;
}
.toolbar .btn {
  padding: 2px 8px;
}
.toolbar .btn-group {
  margin-top: 0px;
  margin-left: 5px;
}
#maintoolbar {
  margin-bottom: -3px;
  margin-top: -8px;
  border: 0px;
  min-height: 27px;
  margin-left: 0px;
  padding-top: 11px;
  padding-bottom: 3px;
}
#maintoolbar .navbar-text {
  float: none;
  vertical-align: middle;
  text-align: right;
  margin-left: 5px;
  margin-right: 0px;
  margin-top: 0px;
}
.select-xs {
  height: 24px;
}
.pulse,
.dropdown-menu > li > a.pulse,
li.pulse > a.dropdown-toggle,
li.pulse.open > a.dropdown-toggle {
  background-color: #F37626;
  color: white;
}
/**
 * Primary styles
 *
 * Author: Jupyter Development Team
 */
/** WARNING IF YOU ARE EDITTING THIS FILE, if this is a .css file, It has a lot
 * of chance of beeing generated from the ../less/[samename].less file, you can
 * try to get back the less file by reverting somme commit in history
 **/
/*
 * We'll try to get something pretty, so we
 * have some strange css to have the scroll bar on
 * the left with fix button on the top right of the tooltip
 */
@-moz-keyframes fadeOut {
  from {
    opacity: 1;
  }
  to {
    opacity: 0;
  }
}
@-webkit-keyframes fadeOut {
  from {
    opacity: 1;
  }
  to {
    opacity: 0;
  }
}
@-moz-keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}
@-webkit-keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}
/*properties of tooltip after "expand"*/
.bigtooltip {
  overflow: auto;
  height: 200px;
  -webkit-transition-property: height;
  -webkit-transition-duration: 500ms;
  -moz-transition-property: height;
  -moz-transition-duration: 500ms;
  transition-property: height;
  transition-duration: 500ms;
}
/*properties of tooltip before "expand"*/
.smalltooltip {
  -webkit-transition-property: height;
  -webkit-transition-duration: 500ms;
  -moz-transition-property: height;
  -moz-transition-duration: 500ms;
  transition-property: height;
  transition-duration: 500ms;
  text-overflow: ellipsis;
  overflow: hidden;
  height: 80px;
}
.tooltipbuttons {
  position: absolute;
  padding-right: 15px;
  top: 0px;
  right: 0px;
}
.tooltiptext {
  /*avoid the button to overlap on some docstring*/
  padding-right: 30px;
}
.ipython_tooltip {
  max-width: 700px;
  /*fade-in animation when inserted*/
  -webkit-animation: fadeOut 400ms;
  -moz-animation: fadeOut 400ms;
  animation: fadeOut 400ms;
  -webkit-animation: fadeIn 400ms;
  -moz-animation: fadeIn 400ms;
  animation: fadeIn 400ms;
  vertical-align: middle;
  background-color: #f7f7f7;
  overflow: visible;
  border: #ababab 1px solid;
  outline: none;
  padding: 3px;
  margin: 0px;
  padding-left: 7px;
  font-family: monospace;
  min-height: 50px;
  -moz-box-shadow: 0px 6px 10px -1px #adadad;
  -webkit-box-shadow: 0px 6px 10px -1px #adadad;
  box-shadow: 0px 6px 10px -1px #adadad;
  border-radius: 2px;
  position: absolute;
  z-index: 1000;
}
.ipython_tooltip a {
  float: right;
}
.ipython_tooltip .tooltiptext pre {
  border: 0;
  border-radius: 0;
  font-size: 100%;
  background-color: #f7f7f7;
}
.pretooltiparrow {
  left: 0px;
  margin: 0px;
  top: -16px;
  width: 40px;
  height: 16px;
  overflow: hidden;
  position: absolute;
}
.pretooltiparrow:before {
  background-color: #f7f7f7;
  border: 1px #ababab solid;
  z-index: 11;
  content: "";
  position: absolute;
  left: 15px;
  top: 10px;
  width: 25px;
  height: 25px;
  -webkit-transform: rotate(45deg);
  -moz-transform: rotate(45deg);
  -ms-transform: rotate(45deg);
  -o-transform: rotate(45deg);
}
ul.typeahead-list i {
  margin-left: -10px;
  width: 18px;
}
ul.typeahead-list {
  max-height: 80vh;
  overflow: auto;
}
ul.typeahead-list > li > a {
  /** Firefox bug **/
  /* see https://github.com/jupyter/notebook/issues/559 */
  white-space: normal;
}
.cmd-palette .modal-body {
  padding: 7px;
}
.cmd-palette form {
  background: white;
}
.cmd-palette input {
  outline: none;
}
.no-shortcut {
  display: none;
}
.command-shortcut:before {
  content: "(command)";
  padding-right: 3px;
  color: #777777;
}
.edit-shortcut:before {
  content: "(edit)";
  padding-right: 3px;
  color: #777777;
}
#find-and-replace #replace-preview .match,
#find-and-replace #replace-preview .insert {
  background-color: #BBDEFB;
  border-color: #90CAF9;
  border-style: solid;
  border-width: 1px;
  border-radius: 0px;
}
#find-and-replace #replace-preview .replace .match {
  background-color: #FFCDD2;
  border-color: #EF9A9A;
  border-radius: 0px;
}
#find-and-replace #replace-preview .replace .insert {
  background-color: #C8E6C9;
  border-color: #A5D6A7;
  border-radius: 0px;
}
#find-and-replace #replace-preview {
  max-height: 60vh;
  overflow: auto;
}
#find-and-replace #replace-preview pre {
  padding: 5px 10px;
}
.terminal-app {
  background: #EEE;
}
.terminal-app #header {
  background: #fff;
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
}
.terminal-app .terminal {
  float: left;
  font-family: monospace;
  color: white;
  background: black;
  padding: 0.4em;
  border-radius: 2px;
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.4);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.4);
}
.terminal-app .terminal,
.terminal-app .terminal dummy-screen {
  line-height: 1em;
  font-size: 14px;
}
.terminal-app .terminal-cursor {
  color: black;
  background: white;
}
.terminal-app #terminado-container {
  margin-top: 20px;
}
/*# sourceMappingURL=style.min.css.map */
    </style>
<style type="text/css">
    .highlight .hll { background-color: #ffffcc }
.highlight  { background: #f8f8f8; }
.highlight .c { color: #408080; font-style: italic } /* Comment */
.highlight .err { border: 1px solid #FF0000 } /* Error */
.highlight .k { color: #008000; font-weight: bold } /* Keyword */
.highlight .o { color: #666666 } /* Operator */
.highlight .ch { color: #408080; font-style: italic } /* Comment.Hashbang */
.highlight .cm { color: #408080; font-style: italic } /* Comment.Multiline */
.highlight .cp { color: #BC7A00 } /* Comment.Preproc */
.highlight .cpf { color: #408080; font-style: italic } /* Comment.PreprocFile */
.highlight .c1 { color: #408080; font-style: italic } /* Comment.Single */
.highlight .cs { color: #408080; font-style: italic } /* Comment.Special */
.highlight .gd { color: #A00000 } /* Generic.Deleted */
.highlight .ge { font-style: italic } /* Generic.Emph */
.highlight .gr { color: #FF0000 } /* Generic.Error */
.highlight .gh { color: #000080; font-weight: bold } /* Generic.Heading */
.highlight .gi { color: #00A000 } /* Generic.Inserted */
.highlight .go { color: #888888 } /* Generic.Output */
.highlight .gp { color: #000080; font-weight: bold } /* Generic.Prompt */
.highlight .gs { font-weight: bold } /* Generic.Strong */
.highlight .gu { color: #800080; font-weight: bold } /* Generic.Subheading */
.highlight .gt { color: #0044DD } /* Generic.Traceback */
.highlight .kc { color: #008000; font-weight: bold } /* Keyword.Constant */
.highlight .kd { color: #008000; font-weight: bold } /* Keyword.Declaration */
.highlight .kn { color: #008000; font-weight: bold } /* Keyword.Namespace */
.highlight .kp { color: #008000 } /* Keyword.Pseudo */
.highlight .kr { color: #008000; font-weight: bold } /* Keyword.Reserved */
.highlight .kt { color: #B00040 } /* Keyword.Type */
.highlight .m { color: #666666 } /* Literal.Number */
.highlight .s { color: #BA2121 } /* Literal.String */
.highlight .na { color: #7D9029 } /* Name.Attribute */
.highlight .nb { color: #008000 } /* Name.Builtin */
.highlight .nc { color: #0000FF; font-weight: bold } /* Name.Class */
.highlight .no { color: #880000 } /* Name.Constant */
.highlight .nd { color: #AA22FF } /* Name.Decorator */
.highlight .ni { color: #999999; font-weight: bold } /* Name.Entity */
.highlight .ne { color: #D2413A; font-weight: bold } /* Name.Exception */
.highlight .nf { color: #0000FF } /* Name.Function */
.highlight .nl { color: #A0A000 } /* Name.Label */
.highlight .nn { color: #0000FF; font-weight: bold } /* Name.Namespace */
.highlight .nt { color: #008000; font-weight: bold } /* Name.Tag */
.highlight .nv { color: #19177C } /* Name.Variable */
.highlight .ow { color: #AA22FF; font-weight: bold } /* Operator.Word */
.highlight .w { color: #bbbbbb } /* Text.Whitespace */
.highlight .mb { color: #666666 } /* Literal.Number.Bin */
.highlight .mf { color: #666666 } /* Literal.Number.Float */
.highlight .mh { color: #666666 } /* Literal.Number.Hex */
.highlight .mi { color: #666666 } /* Literal.Number.Integer */
.highlight .mo { color: #666666 } /* Literal.Number.Oct */
.highlight .sb { color: #BA2121 } /* Literal.String.Backtick */
.highlight .sc { color: #BA2121 } /* Literal.String.Char */
.highlight .sd { color: #BA2121; font-style: italic } /* Literal.String.Doc */
.highlight .s2 { color: #BA2121 } /* Literal.String.Double */
.highlight .se { color: #BB6622; font-weight: bold } /* Literal.String.Escape */
.highlight .sh { color: #BA2121 } /* Literal.String.Heredoc */
.highlight .si { color: #BB6688; font-weight: bold } /* Literal.String.Interpol */
.highlight .sx { color: #008000 } /* Literal.String.Other */
.highlight .sr { color: #BB6688 } /* Literal.String.Regex */
.highlight .s1 { color: #BA2121 } /* Literal.String.Single */
.highlight .ss { color: #19177C } /* Literal.String.Symbol */
.highlight .bp { color: #008000 } /* Name.Builtin.Pseudo */
.highlight .vc { color: #19177C } /* Name.Variable.Class */
.highlight .vg { color: #19177C } /* Name.Variable.Global */
.highlight .vi { color: #19177C } /* Name.Variable.Instance */
.highlight .il { color: #666666 } /* Literal.Number.Integer.Long */
    </style>
<style type="text/css">
    
/* Temporary definitions which will become obsolete with Notebook release 5.0 */
.ansi-black-fg { color: #3E424D; }
.ansi-black-bg { background-color: #3E424D; }
.ansi-black-intense-fg { color: #282C36; }
.ansi-black-intense-bg { background-color: #282C36; }
.ansi-red-fg { color: #E75C58; }
.ansi-red-bg { background-color: #E75C58; }
.ansi-red-intense-fg { color: #B22B31; }
.ansi-red-intense-bg { background-color: #B22B31; }
.ansi-green-fg { color: #00A250; }
.ansi-green-bg { background-color: #00A250; }
.ansi-green-intense-fg { color: #007427; }
.ansi-green-intense-bg { background-color: #007427; }
.ansi-yellow-fg { color: #DDB62B; }
.ansi-yellow-bg { background-color: #DDB62B; }
.ansi-yellow-intense-fg { color: #B27D12; }
.ansi-yellow-intense-bg { background-color: #B27D12; }
.ansi-blue-fg { color: #208FFB; }
.ansi-blue-bg { background-color: #208FFB; }
.ansi-blue-intense-fg { color: #0065CA; }
.ansi-blue-intense-bg { background-color: #0065CA; }
.ansi-magenta-fg { color: #D160C4; }
.ansi-magenta-bg { background-color: #D160C4; }
.ansi-magenta-intense-fg { color: #A03196; }
.ansi-magenta-intense-bg { background-color: #A03196; }
.ansi-cyan-fg { color: #60C6C8; }
.ansi-cyan-bg { background-color: #60C6C8; }
.ansi-cyan-intense-fg { color: #258F8F; }
.ansi-cyan-intense-bg { background-color: #258F8F; }
.ansi-white-fg { color: #C5C1B4; }
.ansi-white-bg { background-color: #C5C1B4; }
.ansi-white-intense-fg { color: #A1A6B2; }
.ansi-white-intense-bg { background-color: #A1A6B2; }

.ansi-bold { font-weight: bold; }

    </style>


<style type="text/css">
/* Overrides of notebook CSS for static HTML export */
body {
  overflow: visible;
  padding: 8px;
}

div#notebook {
  overflow: visible;
  border-top: none;
}

@media print {
  div.cell {
    display: block;
    page-break-inside: avoid;
  } 
  div.output_wrapper { 
    display: block;
    page-break-inside: avoid; 
  }
  div.output { 
    display: block;
    page-break-inside: avoid; 
  }
}
</style>

<!-- Custom stylesheet, it must be in the same directory as the html file -->
<link rel="stylesheet" href="custom.css">

<!-- Loading mathjax macro -->
<!-- Load mathjax -->
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"></script>
    <!-- MathJax configuration -->
    <script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        tex2jax: {
            inlineMath: [ ['$','$'], ["\\(","\\)"] ],
            displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
            processEscapes: true,
            processEnvironments: true
        },
        // Center justify equations in code and markdown cells. Elsewhere
        // we use CSS to left justify single line equations in code cells.
        displayAlign: 'center',
        "HTML-CSS": {
            styles: {'.MathJax_Display': {"margin": 0}},
            linebreaks: { automatic: true }
        }
    });
    </script>
    <!-- End of mathjax configuration --></head>
<body>
  <div tabindex="-1" id="notebook" class="border-box-sizing">
    <div class="container" id="notebook-container">

<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Introduction">Introduction<a class="anchor-link" href="#Introduction">&#182;</a></h1><p>Here we are trying to predict the overall stock market trend by sentiment analysis of 'top 25 news headlines' from 2008-08-11 to 2016-07-01.<br> We built various machine learning models on this data and tried to predict the trend.<br><br>
The datasets used for this projects are:</p>
<ol>
<li>#### Reddit news data: 
Historical news headlines from Reddit News Channel. They are ranked by the reddit user's votes.</li>
<li>#### Stock data: 
Dow Jones Industrial Average (DJIA) is used as the stock market data.</li>
</ol>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Let's import some relevant packages.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[38]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre><span></span><span class="c1"># Load in the relevant libraries</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="kn">as</span> <span class="nn">pd</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="kn">as</span> <span class="nn">plt</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="kn">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">itertools</span>

<span class="kn">from</span> <span class="nn">sklearn.ensemble</span> <span class="kn">import</span> <span class="n">RandomForestClassifier</span>
<span class="kn">from</span> <span class="nn">sklearn.ensemble</span> <span class="kn">import</span> <span class="n">GradientBoostingClassifier</span>
<span class="kn">from</span> <span class="nn">sklearn.ensemble</span> <span class="kn">import</span> <span class="n">AdaBoostClassifier</span>
<span class="kn">from</span> <span class="nn">sklearn.neural_network</span> <span class="kn">import</span> <span class="n">MLPClassifier</span>
<span class="kn">from</span> <span class="nn">sklearn</span> <span class="kn">import</span> <span class="n">linear_model</span>
<span class="kn">from</span> <span class="nn">sklearn.metrics</span> <span class="kn">import</span> <span class="n">confusion_matrix</span>
<span class="kn">from</span> <span class="nn">sklearn.metrics</span> <span class="kn">import</span> <span class="n">classification_report</span>
<span class="kn">from</span> <span class="nn">sklearn.metrics</span> <span class="kn">import</span> <span class="n">accuracy_score</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="1.-Data-transformation:">1. Data transformation:<a class="anchor-link" href="#1.-Data-transformation:">&#182;</a></h3><p>Load in our news data. The sentiment scores for the news is obtained using Stanford NLP Software.</p>
<p>Here we are adding the sentiment scores of 5 consecutive days using 'rolling window'. This will be
the basis of next day stock trend.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[39]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre><span></span><span class="c1"># Load in setiment score&#39;s dataset</span>
<span class="n">df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s2">&quot;Sent_scores.csv&quot;</span><span class="p">,</span> <span class="n">index_col</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>

<span class="c1"># Add 5 consecutive day&#39;s scores</span>
<span class="n">df</span><span class="p">[</span><span class="s2">&quot;PosCount_cv&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="s2">&quot;PosCount&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">rolling</span><span class="p">(</span><span class="n">window</span><span class="o">=</span><span class="mi">5</span><span class="p">)</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>
<span class="n">df</span><span class="p">[</span><span class="s2">&quot;NegCount_cv&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="s2">&quot;NegCount&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">rolling</span><span class="p">(</span><span class="n">window</span><span class="o">=</span><span class="mi">5</span><span class="p">)</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>
<span class="n">df</span><span class="p">[</span><span class="s2">&quot;TrustCount_cv&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="s2">&quot;TrustCount&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">rolling</span><span class="p">(</span><span class="n">window</span><span class="o">=</span><span class="mi">5</span><span class="p">)</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>
<span class="n">df</span><span class="p">[</span><span class="s2">&quot;AngerCount_cv&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="s2">&quot;AngerCount&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">rolling</span><span class="p">(</span><span class="n">window</span><span class="o">=</span><span class="mi">5</span><span class="p">)</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>
<span class="n">df</span><span class="p">[</span><span class="s2">&quot;AnticipationCount_cv&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="s2">&quot;AnticipationCount&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">rolling</span><span class="p">(</span><span class="n">window</span><span class="o">=</span><span class="mi">5</span><span class="p">)</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>
<span class="n">df</span><span class="p">[</span><span class="s2">&quot;DisgustCount_cv&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="s2">&quot;DisgustCount&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">rolling</span><span class="p">(</span><span class="n">window</span><span class="o">=</span><span class="mi">5</span><span class="p">)</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>
<span class="n">df</span><span class="p">[</span><span class="s2">&quot;FearCount_cv&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="s2">&quot;FearCount&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">rolling</span><span class="p">(</span><span class="n">window</span><span class="o">=</span><span class="mi">5</span><span class="p">)</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>
<span class="n">df</span><span class="p">[</span><span class="s2">&quot;JoyCount_cv&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="s2">&quot;JoyCount&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">rolling</span><span class="p">(</span><span class="n">window</span><span class="o">=</span><span class="mi">5</span><span class="p">)</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>
<span class="n">df</span><span class="p">[</span><span class="s2">&quot;SadnessCount_cv&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="s2">&quot;SadnessCount&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">rolling</span><span class="p">(</span><span class="n">window</span><span class="o">=</span><span class="mi">5</span><span class="p">)</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>
<span class="n">df</span><span class="p">[</span><span class="s2">&quot;SurpriseCount_cv&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="s2">&quot;SurpriseCount&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">rolling</span><span class="p">(</span><span class="n">window</span><span class="o">=</span><span class="mi">5</span><span class="p">)</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>
<span class="n">df</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">5</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[39]:</div>

<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>PosCount</th>
      <th>NegCount</th>
      <th>TrustCount</th>
      <th>AngerCount</th>
      <th>AnticipationCount</th>
      <th>DisgustCount</th>
      <th>FearCount</th>
      <th>JoyCount</th>
      <th>SadnessCount</th>
      <th>...</th>
      <th>PosCount_cv</th>
      <th>NegCount_cv</th>
      <th>TrustCount_cv</th>
      <th>AngerCount_cv</th>
      <th>AnticipationCount_cv</th>
      <th>DisgustCount_cv</th>
      <th>FearCount_cv</th>
      <th>JoyCount_cv</th>
      <th>SadnessCount_cv</th>
      <th>SurpriseCount_cv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2008-06-08</td>
      <td>19</td>
      <td>28</td>
      <td>12</td>
      <td>15</td>
      <td>9</td>
      <td>5</td>
      <td>22</td>
      <td>3</td>
      <td>10</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2008-06-09</td>
      <td>25</td>
      <td>25</td>
      <td>20</td>
      <td>16</td>
      <td>16</td>
      <td>3</td>
      <td>22</td>
      <td>9</td>
      <td>9</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2008-06-10</td>
      <td>11</td>
      <td>27</td>
      <td>12</td>
      <td>16</td>
      <td>12</td>
      <td>9</td>
      <td>21</td>
      <td>6</td>
      <td>12</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2008-06-11</td>
      <td>19</td>
      <td>19</td>
      <td>15</td>
      <td>11</td>
      <td>6</td>
      <td>9</td>
      <td>15</td>
      <td>7</td>
      <td>6</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2008-06-12</td>
      <td>17</td>
      <td>24</td>
      <td>15</td>
      <td>15</td>
      <td>8</td>
      <td>10</td>
      <td>20</td>
      <td>6</td>
      <td>12</td>
      <td>...</td>
      <td>91.0</td>
      <td>123.0</td>
      <td>74.0</td>
      <td>73.0</td>
      <td>51.0</td>
      <td>36.0</td>
      <td>100.0</td>
      <td>31.0</td>
      <td>49.0</td>
      <td>18.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows � 21 columns</p>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[30]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre><span></span><span class="c1"># Load in stock market data</span>
<span class="n">stock_df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s2">&quot;DJIA_table.csv&quot;</span><span class="p">,</span> <span class="n">index_col</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">stock_df</span> <span class="o">=</span> <span class="n">stock_df</span><span class="o">.</span><span class="n">iloc</span><span class="p">[::</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span>
<span class="n">stock_df</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">5</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[30]:</div>

<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
    </tr>
    <tr>
      <th>Open</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11432.089844</th>
      <td>2008-08-08</td>
      <td>11759.959961</td>
      <td>11388.040039</td>
      <td>11734.320312</td>
      <td>212830000</td>
      <td>11734.320312</td>
    </tr>
    <tr>
      <th>11729.669922</th>
      <td>2008-08-11</td>
      <td>11867.110352</td>
      <td>11675.530273</td>
      <td>11782.349609</td>
      <td>183190000</td>
      <td>11782.349609</td>
    </tr>
    <tr>
      <th>11781.700195</th>
      <td>2008-08-12</td>
      <td>11782.349609</td>
      <td>11601.519531</td>
      <td>11642.469727</td>
      <td>173590000</td>
      <td>11642.469727</td>
    </tr>
    <tr>
      <th>11632.809570</th>
      <td>2008-08-13</td>
      <td>11633.780273</td>
      <td>11453.339844</td>
      <td>11532.959961</td>
      <td>182550000</td>
      <td>11532.959961</td>
    </tr>
    <tr>
      <th>11532.070312</th>
      <td>2008-08-14</td>
      <td>11718.280273</td>
      <td>11450.889648</td>
      <td>11615.929688</td>
      <td>159790000</td>
      <td>11615.929688</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Merge the two datasets and drop the irrelevant variables. Also, assign the binary class to each day stock trend. If today's stock price is larger than previous day's then assign class '1' otherwise assign class '0'.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[40]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre><span></span><span class="c1"># Mearge two datasets</span>
<span class="n">df1</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">merge</span><span class="p">(</span><span class="n">df</span><span class="p">,</span> <span class="n">stock_df</span><span class="p">)</span>

<span class="c1"># Drop irrelevant variabls</span>
<span class="n">df2</span> <span class="o">=</span> <span class="n">df1</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="s1">&#39;Volume&#39;</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">df2</span> <span class="o">=</span> <span class="n">df2</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="s1">&#39;High&#39;</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">df2</span> <span class="o">=</span> <span class="n">df2</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="s1">&#39;Low&#39;</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">df2</span> <span class="o">=</span> <span class="n">df2</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="s1">&#39;Close&#39;</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="c1">#df2 = df2.drop(&#39;Date&#39;, axis=1)</span>
<span class="n">df2</span><span class="p">[</span><span class="s2">&quot;trend&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="mi">0</span>

<span class="c1"># Assigning class to each day stock trend</span>
<span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="nb">len</span><span class="p">(</span><span class="n">df2</span><span class="p">[</span><span class="s2">&quot;Adj Close&quot;</span><span class="p">])):</span>
    <span class="k">if</span><span class="p">(</span><span class="n">df2</span><span class="p">[</span><span class="s2">&quot;Adj Close&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">iloc</span><span class="p">[</span><span class="n">i</span><span class="p">]</span><span class="o">&gt;=</span><span class="n">df2</span><span class="p">[</span><span class="s2">&quot;Adj Close&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">iloc</span><span class="p">[</span><span class="n">i</span><span class="o">-</span><span class="mi">1</span><span class="p">]):</span>
        <span class="n">df2</span><span class="p">[</span><span class="s2">&quot;trend&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">iloc</span><span class="p">[</span><span class="n">i</span><span class="p">]</span><span class="o">=</span><span class="mi">1</span>
    <span class="k">else</span><span class="p">:</span>
        <span class="n">df2</span><span class="p">[</span><span class="s2">&quot;trend&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">iloc</span><span class="p">[</span><span class="n">i</span><span class="p">]</span><span class="o">=</span><span class="mi">0</span>

<span class="n">df2</span> <span class="o">=</span> <span class="n">df2</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="n">df2</span><span class="o">.</span><span class="n">index</span><span class="p">[[</span><span class="mi">0</span><span class="p">]])</span>
<span class="n">df2</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">5</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[40]:</div>

<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>PosCount</th>
      <th>NegCount</th>
      <th>TrustCount</th>
      <th>AngerCount</th>
      <th>AnticipationCount</th>
      <th>DisgustCount</th>
      <th>FearCount</th>
      <th>JoyCount</th>
      <th>SadnessCount</th>
      <th>...</th>
      <th>TrustCount_cv</th>
      <th>AngerCount_cv</th>
      <th>AnticipationCount_cv</th>
      <th>DisgustCount_cv</th>
      <th>FearCount_cv</th>
      <th>JoyCount_cv</th>
      <th>SadnessCount_cv</th>
      <th>SurpriseCount_cv</th>
      <th>Adj Close</th>
      <th>trend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2008-08-11</td>
      <td>12</td>
      <td>15</td>
      <td>7</td>
      <td>8</td>
      <td>7</td>
      <td>3</td>
      <td>15</td>
      <td>5</td>
      <td>6</td>
      <td>...</td>
      <td>51.0</td>
      <td>48.0</td>
      <td>33.0</td>
      <td>20.0</td>
      <td>87.0</td>
      <td>19.0</td>
      <td>37.0</td>
      <td>22.0</td>
      <td>11782.349609</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2008-08-12</td>
      <td>12</td>
      <td>21</td>
      <td>6</td>
      <td>10</td>
      <td>3</td>
      <td>1</td>
      <td>16</td>
      <td>4</td>
      <td>8</td>
      <td>...</td>
      <td>41.0</td>
      <td>47.0</td>
      <td>28.0</td>
      <td>15.0</td>
      <td>84.0</td>
      <td>20.0</td>
      <td>37.0</td>
      <td>20.0</td>
      <td>11642.469727</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2008-08-13</td>
      <td>21</td>
      <td>19</td>
      <td>19</td>
      <td>12</td>
      <td>5</td>
      <td>2</td>
      <td>16</td>
      <td>6</td>
      <td>8</td>
      <td>...</td>
      <td>52.0</td>
      <td>45.0</td>
      <td>26.0</td>
      <td>9.0</td>
      <td>78.0</td>
      <td>24.0</td>
      <td>36.0</td>
      <td>19.0</td>
      <td>11532.959961</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2008-08-14</td>
      <td>22</td>
      <td>21</td>
      <td>16</td>
      <td>12</td>
      <td>15</td>
      <td>3</td>
      <td>17</td>
      <td>9</td>
      <td>10</td>
      <td>...</td>
      <td>59.0</td>
      <td>52.0</td>
      <td>38.0</td>
      <td>11.0</td>
      <td>81.0</td>
      <td>29.0</td>
      <td>42.0</td>
      <td>23.0</td>
      <td>11615.929688</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2008-08-15</td>
      <td>15</td>
      <td>23</td>
      <td>14</td>
      <td>14</td>
      <td>4</td>
      <td>3</td>
      <td>20</td>
      <td>8</td>
      <td>8</td>
      <td>...</td>
      <td>62.0</td>
      <td>56.0</td>
      <td>34.0</td>
      <td>12.0</td>
      <td>84.0</td>
      <td>32.0</td>
      <td>40.0</td>
      <td>22.0</td>
      <td>11659.900391</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows � 23 columns</p>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[41]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre><span></span><span class="n">df2</span> <span class="o">=</span> <span class="n">df2</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="s1">&#39;Adj Close&#39;</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">df2</span><span class="p">[</span><span class="s2">&quot;trend&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">5</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[41]:</div>


<div class="output_text output_subarea output_execute_result">
<pre>1    1
2    0
3    0
4    1
5    1
Name: trend, dtype: int64</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Train-and-Test-split:">Train and Test split:<a class="anchor-link" href="#Train-and-Test-split:">&#182;</a></h4><p>Split the 70% data as a training set and other 30% as a test set which will be used for predicting stock market trend.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[42]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre><span></span><span class="c1"># train-test split</span>
<span class="n">train</span> <span class="o">=</span> <span class="n">df2</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="mi">0</span><span class="p">:</span><span class="mi">1392</span><span class="p">,:]</span>
<span class="n">test</span> <span class="o">=</span> <span class="n">df2</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="n">train</span><span class="o">.</span><span class="n">index</span><span class="p">)</span>

<span class="n">date_train</span> <span class="o">=</span> <span class="n">train</span><span class="p">[</span><span class="s2">&quot;Date&quot;</span><span class="p">]</span>
<span class="n">date_test</span> <span class="o">=</span> <span class="n">test</span><span class="p">[</span><span class="s2">&quot;Date&quot;</span><span class="p">]</span>

<span class="n">train</span> <span class="o">=</span> <span class="n">train</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="s1">&#39;Date&#39;</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">test</span> <span class="o">=</span> <span class="n">test</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="s1">&#39;Date&#39;</span><span class="p">,</span><span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>

<span class="n">X</span> <span class="o">=</span> <span class="n">train</span><span class="o">.</span><span class="n">loc</span><span class="p">[:,</span> <span class="n">train</span><span class="o">.</span><span class="n">columns</span><span class="o">!=</span><span class="s2">&quot;trend&quot;</span><span class="p">]</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">train</span><span class="o">.</span><span class="n">iloc</span><span class="p">[:,</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span>

<span class="n">test_X</span> <span class="o">=</span> <span class="n">test</span><span class="o">.</span><span class="n">loc</span><span class="p">[:,</span> <span class="n">test</span><span class="o">.</span><span class="n">columns</span><span class="o">!=</span><span class="s2">&quot;trend&quot;</span><span class="p">]</span>
<span class="n">test_y</span> <span class="o">=</span> <span class="n">test</span><span class="o">.</span><span class="n">iloc</span><span class="p">[:,</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span>

<span class="n">train</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">5</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[42]:</div>

<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PosCount</th>
      <th>NegCount</th>
      <th>TrustCount</th>
      <th>AngerCount</th>
      <th>AnticipationCount</th>
      <th>DisgustCount</th>
      <th>FearCount</th>
      <th>JoyCount</th>
      <th>SadnessCount</th>
      <th>SurpriseCount</th>
      <th>...</th>
      <th>NegCount_cv</th>
      <th>TrustCount_cv</th>
      <th>AngerCount_cv</th>
      <th>AnticipationCount_cv</th>
      <th>DisgustCount_cv</th>
      <th>FearCount_cv</th>
      <th>JoyCount_cv</th>
      <th>SadnessCount_cv</th>
      <th>SurpriseCount_cv</th>
      <th>trend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>12</td>
      <td>15</td>
      <td>7</td>
      <td>8</td>
      <td>7</td>
      <td>3</td>
      <td>15</td>
      <td>5</td>
      <td>6</td>
      <td>5</td>
      <td>...</td>
      <td>99.0</td>
      <td>51.0</td>
      <td>48.0</td>
      <td>33.0</td>
      <td>20.0</td>
      <td>87.0</td>
      <td>19.0</td>
      <td>37.0</td>
      <td>22.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12</td>
      <td>21</td>
      <td>6</td>
      <td>10</td>
      <td>3</td>
      <td>1</td>
      <td>16</td>
      <td>4</td>
      <td>8</td>
      <td>3</td>
      <td>...</td>
      <td>95.0</td>
      <td>41.0</td>
      <td>47.0</td>
      <td>28.0</td>
      <td>15.0</td>
      <td>84.0</td>
      <td>20.0</td>
      <td>37.0</td>
      <td>20.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21</td>
      <td>19</td>
      <td>19</td>
      <td>12</td>
      <td>5</td>
      <td>2</td>
      <td>16</td>
      <td>6</td>
      <td>8</td>
      <td>3</td>
      <td>...</td>
      <td>89.0</td>
      <td>52.0</td>
      <td>45.0</td>
      <td>26.0</td>
      <td>9.0</td>
      <td>78.0</td>
      <td>24.0</td>
      <td>36.0</td>
      <td>19.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22</td>
      <td>21</td>
      <td>16</td>
      <td>12</td>
      <td>15</td>
      <td>3</td>
      <td>17</td>
      <td>9</td>
      <td>10</td>
      <td>8</td>
      <td>...</td>
      <td>96.0</td>
      <td>59.0</td>
      <td>52.0</td>
      <td>38.0</td>
      <td>11.0</td>
      <td>81.0</td>
      <td>29.0</td>
      <td>42.0</td>
      <td>23.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>15</td>
      <td>23</td>
      <td>14</td>
      <td>14</td>
      <td>4</td>
      <td>3</td>
      <td>20</td>
      <td>8</td>
      <td>8</td>
      <td>3</td>
      <td>...</td>
      <td>99.0</td>
      <td>62.0</td>
      <td>56.0</td>
      <td>34.0</td>
      <td>12.0</td>
      <td>84.0</td>
      <td>32.0</td>
      <td>40.0</td>
      <td>22.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows � 21 columns</p>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="2.-Machine-Learning-models">2. Machine Learning models<a class="anchor-link" href="#2.-Machine-Learning-models">&#182;</a></h3><p>Implementing some machine learning models including Random forest, KNN, Neural Networks, Naive Bayes, Decision tree, etc. using scikit-learn library to predict the stock trend.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Random-Forest">Random Forest<a class="anchor-link" href="#Random-Forest">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[56]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre><span></span><span class="n">clf</span> <span class="o">=</span> <span class="n">RandomForestClassifier</span><span class="p">(</span><span class="n">max_depth</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
<span class="c1">#clf = GradientBoostingClassifier(n_estimators=1000, max_depth=10)</span>
<span class="c1">#clf = AdaBoostClassifier(n_estimators=100)</span>
<span class="c1">#clf = linear_model.LinearRegression()</span>

<span class="n">clf</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X</span><span class="p">,</span><span class="n">y</span><span class="p">)</span>
<span class="n">test</span><span class="p">[</span><span class="s2">&quot;predict&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">clf</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">test_X</span><span class="p">)</span>
<span class="n">test</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">5</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[56]:</div>

<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PosCount</th>
      <th>NegCount</th>
      <th>TrustCount</th>
      <th>AngerCount</th>
      <th>AnticipationCount</th>
      <th>DisgustCount</th>
      <th>FearCount</th>
      <th>JoyCount</th>
      <th>SadnessCount</th>
      <th>SurpriseCount</th>
      <th>...</th>
      <th>TrustCount_cv</th>
      <th>AngerCount_cv</th>
      <th>AnticipationCount_cv</th>
      <th>DisgustCount_cv</th>
      <th>FearCount_cv</th>
      <th>JoyCount_cv</th>
      <th>SadnessCount_cv</th>
      <th>SurpriseCount_cv</th>
      <th>trend</th>
      <th>predict</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1393</th>
      <td>30</td>
      <td>35</td>
      <td>27</td>
      <td>20</td>
      <td>9</td>
      <td>10</td>
      <td>31</td>
      <td>6</td>
      <td>17</td>
      <td>9</td>
      <td>...</td>
      <td>99.0</td>
      <td>82.0</td>
      <td>46.0</td>
      <td>44.0</td>
      <td>125.0</td>
      <td>27.0</td>
      <td>70.0</td>
      <td>29.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1394</th>
      <td>25</td>
      <td>33</td>
      <td>20</td>
      <td>20</td>
      <td>12</td>
      <td>5</td>
      <td>29</td>
      <td>10</td>
      <td>14</td>
      <td>3</td>
      <td>...</td>
      <td>107.0</td>
      <td>77.0</td>
      <td>54.0</td>
      <td>38.0</td>
      <td>112.0</td>
      <td>41.0</td>
      <td>63.0</td>
      <td>30.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1395</th>
      <td>23</td>
      <td>25</td>
      <td>23</td>
      <td>13</td>
      <td>12</td>
      <td>5</td>
      <td>11</td>
      <td>10</td>
      <td>10</td>
      <td>7</td>
      <td>...</td>
      <td>109.0</td>
      <td>76.0</td>
      <td>57.0</td>
      <td>36.0</td>
      <td>98.0</td>
      <td>42.0</td>
      <td>59.0</td>
      <td>30.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1396</th>
      <td>23</td>
      <td>22</td>
      <td>10</td>
      <td>9</td>
      <td>5</td>
      <td>6</td>
      <td>15</td>
      <td>4</td>
      <td>11</td>
      <td>2</td>
      <td>...</td>
      <td>92.0</td>
      <td>65.0</td>
      <td>53.0</td>
      <td>32.0</td>
      <td>82.0</td>
      <td>40.0</td>
      <td>53.0</td>
      <td>23.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1397</th>
      <td>27</td>
      <td>22</td>
      <td>19</td>
      <td>13</td>
      <td>11</td>
      <td>3</td>
      <td>18</td>
      <td>8</td>
      <td>9</td>
      <td>4</td>
      <td>...</td>
      <td>85.0</td>
      <td>68.0</td>
      <td>47.0</td>
      <td>27.0</td>
      <td>88.0</td>
      <td>36.0</td>
      <td>52.0</td>
      <td>19.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows � 22 columns</p>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[57]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre><span></span><span class="c1">#confusion_matrix</span>
<span class="n">cnf_matrix</span> <span class="o">=</span> <span class="n">confusion_matrix</span><span class="p">(</span><span class="n">test</span><span class="p">[</span><span class="s2">&quot;trend&quot;</span><span class="p">],</span> <span class="n">test</span><span class="p">[</span><span class="s2">&quot;predict&quot;</span><span class="p">])</span>
<span class="n">np</span><span class="o">.</span><span class="n">set_printoptions</span><span class="p">(</span><span class="n">precision</span><span class="o">=</span><span class="mi">2</span><span class="p">)</span>
<span class="n">class_names</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;1&#39;</span><span class="p">,</span><span class="s1">&#39;0&#39;</span><span class="p">]</span>

<span class="k">def</span> <span class="nf">plot_confusion_matrix</span><span class="p">(</span><span class="n">cm</span><span class="p">,</span> <span class="n">classes</span><span class="p">,</span>
                          <span class="n">normalize</span><span class="o">=</span><span class="bp">False</span><span class="p">,</span>
                          <span class="n">title</span><span class="o">=</span><span class="s1">&#39;Confusion matrix&#39;</span><span class="p">,</span>
                          <span class="n">cmap</span><span class="o">=</span><span class="n">plt</span><span class="o">.</span><span class="n">cm</span><span class="o">.</span><span class="n">Blues</span><span class="p">):</span>
    <span class="sd">&quot;&quot;&quot;</span>
<span class="sd">    This function prints and plots the confusion matrix.</span>
<span class="sd">    Normalization can be applied by setting `normalize=True`.</span>
<span class="sd">    &quot;&quot;&quot;</span>
    <span class="k">if</span> <span class="n">normalize</span><span class="p">:</span>
        <span class="n">cm</span> <span class="o">=</span> <span class="n">cm</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="s1">&#39;float&#39;</span><span class="p">)</span> <span class="o">/</span> <span class="n">cm</span><span class="o">.</span><span class="n">sum</span><span class="p">(</span><span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)[:,</span> <span class="n">np</span><span class="o">.</span><span class="n">newaxis</span><span class="p">]</span>
        <span class="k">print</span><span class="p">(</span><span class="s2">&quot;Normalized confusion matrix&quot;</span><span class="p">)</span>
    <span class="k">else</span><span class="p">:</span>
        <span class="k">print</span><span class="p">(</span><span class="s1">&#39;Confusion matrix&#39;</span><span class="p">)</span>

    <span class="k">print</span><span class="p">(</span><span class="n">cm</span><span class="p">)</span>

    <span class="n">plt</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">cm</span><span class="p">,</span> <span class="n">interpolation</span><span class="o">=</span><span class="s1">&#39;nearest&#39;</span><span class="p">,</span> <span class="n">cmap</span><span class="o">=</span><span class="n">cmap</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="n">title</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">colorbar</span><span class="p">()</span>
    <span class="n">tick_marks</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">arange</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">classes</span><span class="p">))</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">xticks</span><span class="p">(</span><span class="n">tick_marks</span><span class="p">,</span> <span class="n">classes</span><span class="p">,</span> <span class="n">rotation</span><span class="o">=</span><span class="mi">45</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">yticks</span><span class="p">(</span><span class="n">tick_marks</span><span class="p">,</span> <span class="n">classes</span><span class="p">)</span>

    <span class="n">fmt</span> <span class="o">=</span> <span class="s1">&#39;.2f&#39;</span> <span class="k">if</span> <span class="n">normalize</span> <span class="k">else</span> <span class="s1">&#39;d&#39;</span>
    <span class="n">thresh</span> <span class="o">=</span> <span class="n">cm</span><span class="o">.</span><span class="n">max</span><span class="p">()</span> <span class="o">/</span> <span class="mf">2.</span>
    <span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="n">j</span> <span class="ow">in</span> <span class="n">itertools</span><span class="o">.</span><span class="n">product</span><span class="p">(</span><span class="nb">range</span><span class="p">(</span><span class="n">cm</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">]),</span> <span class="nb">range</span><span class="p">(</span><span class="n">cm</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">1</span><span class="p">])):</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">text</span><span class="p">(</span><span class="n">j</span><span class="p">,</span> <span class="n">i</span><span class="p">,</span> <span class="n">format</span><span class="p">(</span><span class="n">cm</span><span class="p">[</span><span class="n">i</span><span class="p">,</span> <span class="n">j</span><span class="p">],</span> <span class="n">fmt</span><span class="p">),</span>
                 <span class="n">horizontalalignment</span><span class="o">=</span><span class="s2">&quot;center&quot;</span><span class="p">,</span>
                 <span class="n">color</span><span class="o">=</span><span class="s2">&quot;white&quot;</span> <span class="k">if</span> <span class="n">cm</span><span class="p">[</span><span class="n">i</span><span class="p">,</span> <span class="n">j</span><span class="p">]</span> <span class="o">&gt;</span> <span class="n">thresh</span> <span class="k">else</span> <span class="s2">&quot;black&quot;</span><span class="p">)</span>

    <span class="n">plt</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">&#39;True label&#39;</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s1">&#39;Predicted label&#39;</span><span class="p">)</span>
<span class="c1"># Plot non-normalized confusion matrix</span>
<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">()</span>
<span class="n">plot_confusion_matrix</span><span class="p">(</span><span class="n">cnf_matrix</span><span class="p">,</span> <span class="n">classes</span><span class="o">=</span><span class="n">class_names</span><span class="p">,</span>
                      <span class="n">title</span><span class="o">=</span><span class="s1">&#39;Confusion matrix&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>Confusion matrix
[[ 20 258]
 [ 24 294]]
</pre>
</div>
</div>

<div class="output_area"><div class="prompt"></div>


<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[58]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre><span></span><span class="k">print</span><span class="p">(</span><span class="n">accuracy_score</span><span class="p">(</span><span class="n">test</span><span class="p">[</span><span class="s2">&quot;trend&quot;</span><span class="p">],</span> <span class="n">test</span><span class="p">[</span><span class="s2">&quot;predict&quot;</span><span class="p">]))</span>
<span class="k">print</span><span class="p">(</span><span class="n">confusion_matrix</span><span class="p">(</span><span class="n">test</span><span class="p">[</span><span class="s2">&quot;trend&quot;</span><span class="p">],</span> <span class="n">test</span><span class="p">[</span><span class="s2">&quot;predict&quot;</span><span class="p">]))</span>
<span class="k">print</span><span class="p">(</span><span class="n">classification_report</span><span class="p">(</span><span class="n">test</span><span class="p">[</span><span class="s2">&quot;trend&quot;</span><span class="p">],</span> <span class="n">test</span><span class="p">[</span><span class="s2">&quot;predict&quot;</span><span class="p">]))</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>0.526845637584
[[ 20 258]
 [ 24 294]]
             precision    recall  f1-score   support

          0       0.45      0.07      0.12       278
          1       0.53      0.92      0.68       318

avg / total       0.50      0.53      0.42       596

</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Feature-Importance">Feature Importance<a class="anchor-link" href="#Feature-Importance">&#182;</a></h2><p>As expected, importance of features - 'moving sum sentiment scores' is higher than daywise sentiment scores!</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[59]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre><span></span><span class="c1"># feature_importancefrom sklearn import tree</span>
<span class="n">importances</span> <span class="o">=</span> <span class="n">clf</span><span class="o">.</span><span class="n">feature_importances_</span>
<span class="n">indices</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">argsort</span><span class="p">(</span><span class="n">importances</span><span class="p">)[::</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span>
<span class="n">feature_names</span><span class="o">=</span> <span class="n">X</span><span class="o">.</span><span class="n">columns</span><span class="p">[</span><span class="n">indices</span><span class="p">]</span>
<span class="n">f</span><span class="p">,</span> <span class="n">ax</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">11</span><span class="p">,</span> <span class="mi">9</span><span class="p">))</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">&quot;Feature ranking&quot;</span><span class="p">,</span> <span class="n">fontsize</span> <span class="o">=</span> <span class="mi">20</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">bar</span><span class="p">(</span><span class="nb">range</span><span class="p">(</span><span class="n">X</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">1</span><span class="p">]),</span> <span class="n">importances</span><span class="p">[</span><span class="n">indices</span><span class="p">],</span><span class="n">color</span><span class="o">=</span><span class="s2">&quot;b&quot;</span><span class="p">,</span> <span class="n">align</span><span class="o">=</span><span class="s2">&quot;center&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xticks</span><span class="p">(</span><span class="nb">range</span><span class="p">(</span><span class="n">X</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">1</span><span class="p">]),</span> <span class="n">feature_names</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlim</span><span class="p">([</span><span class="o">-</span><span class="mi">1</span><span class="p">,</span> <span class="n">X</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">1</span><span class="p">]])</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s2">&quot;importance&quot;</span><span class="p">,</span> <span class="n">fontsize</span> <span class="o">=</span> <span class="mi">18</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s2">&quot;index of the feature&quot;</span><span class="p">,</span> <span class="n">fontsize</span> <span class="o">=</span> <span class="mf">0.05</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xticks</span><span class="p">(</span><span class="n">rotation</span><span class="o">=</span><span class="mi">90</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt"></div>


<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
mmmlOhnXWVU/TbIfzfRib0myZVW9bsw1rk/yTJp5YE9LcgrNVFZ3o5k+7HSa+XhvmPtZJPWZe1Yl
9V5VXU1zuPxI4JfA44GjaPYUXt1+/8WB7T9DM5H9d4CDgGcAv2+3P32OMq+mmZLpdsDLgH+hmZx/
5jmPp5mT9Cc0U2s9geYM9H1ZNW5yfV/XN4C7Av9Bc4b7M2iC+V40Fz94CquGIIysnbLqEcBxNHO4
vgC4N034PoBVJy6NXVX9AnggzXCOf0/yTx3U+ALN7/bLNGNzn0+zt/t+rPoAM9LvSNL0pZm6TpKk
jU+Sc2nC/zZVtdA91JKmwD2rkqQNWpKbtCe6DS9/Js3wj88aVKUNl3tWJUkbtCR3prn4wpk088hu
TrM39d40w0b2qarvTq+FkhbCsCpJ2qAl+Uuacb/3p5lX9cY0Y4vPBF5TVd+bYvMkLZBhVZIkSb3l
mFVJkiT11iY1z+oOO+xQu+2227SbIUmStEk7//zzf15Vi+ez7SYVVnfbbTeWLVs27WZIkiRt0pL8
YN1bNRwGIEmSpN4yrEqSJKm3DKuSJEnqLcOqJEmSesuwKkmSpN4yrEqSJKm3DKuSJEnqLcOqJEmS
esuwKkmSpN4yrEqSJKm3DKuSJEnqLcOqJEmSesuwKkmSpN4yrEqSJKm3DKuSJEnqLcOqJEmSesuw
KkmSpN4yrEqSJKm3DKuSJEnqLcOqJEmSesuwKkmSpN7abNoN2NAk3T13VXfPLUmStCEyrG4AugzI
YEiWJEn95TAASZIk9ZZhVZIkSb1lWJUkSVJvGVYlSZLUW4ZVSZIk9ZZhVZIkSb1lWJUkSVJvGVYl
SZLUW4ZVSZIk9ZZhVZIkSb1lWJUkSVJvGVYlSZLUW4ZVSZIk9dZm026A+ivp7rmruntuSZK08XDP
qiRJknrLsCpJkqTeMqxKkiSptwyrkiRJ6i3DqiRJknrLsCpJkqTeMqxKkiSptwyrkiRJ6i3DqiRJ
knrLsCpJkqTeMqxKkiSptwyrkiRJ6i3DqiRJknprs2k3QBqUdPfcVd09tyRJ6oZ7ViVJktRbhlVJ
kiT1lmFVkiRJvWVYlSRJUm8ZViVJktRbhlVJkiT1lmFVkiRJvWVYlSRJUm8ZViVJktRbhlVJkiT1
lmFVkiRJvWVYlSRJUm9NNawm2T/JZUmWJzl6lvV3SHJOkmuTvHhg+a5JzkpySZKLkxw52ZZLkiRp
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
QldzAAAAAElFTkSuQmCC
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Gradient-Boosting-Classifier">Gradient Boosting Classifier<a class="anchor-link" href="#Gradient-Boosting-Classifier">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[61]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre><span></span><span class="n">clf</span> <span class="o">=</span> <span class="n">GradientBoostingClassifier</span><span class="p">(</span><span class="n">n_estimators</span><span class="o">=</span><span class="mi">1000</span><span class="p">,</span> <span class="n">max_depth</span><span class="o">=</span><span class="mi">10</span><span class="p">)</span>
<span class="c1">#clf = AdaBoostClassifier(n_estimators=100)</span>
<span class="c1">#clf = linear_model.LinearRegression()</span>

<span class="n">clf</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X</span><span class="p">,</span><span class="n">y</span><span class="p">)</span>
<span class="n">test</span><span class="p">[</span><span class="s2">&quot;predict&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">clf</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">test_X</span><span class="p">)</span>
<span class="n">test</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">5</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[61]:</div>

<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PosCount</th>
      <th>NegCount</th>
      <th>TrustCount</th>
      <th>AngerCount</th>
      <th>AnticipationCount</th>
      <th>DisgustCount</th>
      <th>FearCount</th>
      <th>JoyCount</th>
      <th>SadnessCount</th>
      <th>SurpriseCount</th>
      <th>...</th>
      <th>TrustCount_cv</th>
      <th>AngerCount_cv</th>
      <th>AnticipationCount_cv</th>
      <th>DisgustCount_cv</th>
      <th>FearCount_cv</th>
      <th>JoyCount_cv</th>
      <th>SadnessCount_cv</th>
      <th>SurpriseCount_cv</th>
      <th>trend</th>
      <th>predict</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1393</th>
      <td>30</td>
      <td>35</td>
      <td>27</td>
      <td>20</td>
      <td>9</td>
      <td>10</td>
      <td>31</td>
      <td>6</td>
      <td>17</td>
      <td>9</td>
      <td>...</td>
      <td>99.0</td>
      <td>82.0</td>
      <td>46.0</td>
      <td>44.0</td>
      <td>125.0</td>
      <td>27.0</td>
      <td>70.0</td>
      <td>29.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1394</th>
      <td>25</td>
      <td>33</td>
      <td>20</td>
      <td>20</td>
      <td>12</td>
      <td>5</td>
      <td>29</td>
      <td>10</td>
      <td>14</td>
      <td>3</td>
      <td>...</td>
      <td>107.0</td>
      <td>77.0</td>
      <td>54.0</td>
      <td>38.0</td>
      <td>112.0</td>
      <td>41.0</td>
      <td>63.0</td>
      <td>30.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1395</th>
      <td>23</td>
      <td>25</td>
      <td>23</td>
      <td>13</td>
      <td>12</td>
      <td>5</td>
      <td>11</td>
      <td>10</td>
      <td>10</td>
      <td>7</td>
      <td>...</td>
      <td>109.0</td>
      <td>76.0</td>
      <td>57.0</td>
      <td>36.0</td>
      <td>98.0</td>
      <td>42.0</td>
      <td>59.0</td>
      <td>30.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1396</th>
      <td>23</td>
      <td>22</td>
      <td>10</td>
      <td>9</td>
      <td>5</td>
      <td>6</td>
      <td>15</td>
      <td>4</td>
      <td>11</td>
      <td>2</td>
      <td>...</td>
      <td>92.0</td>
      <td>65.0</td>
      <td>53.0</td>
      <td>32.0</td>
      <td>82.0</td>
      <td>40.0</td>
      <td>53.0</td>
      <td>23.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1397</th>
      <td>27</td>
      <td>22</td>
      <td>19</td>
      <td>13</td>
      <td>11</td>
      <td>3</td>
      <td>18</td>
      <td>8</td>
      <td>9</td>
      <td>4</td>
      <td>...</td>
      <td>85.0</td>
      <td>68.0</td>
      <td>47.0</td>
      <td>27.0</td>
      <td>88.0</td>
      <td>36.0</td>
      <td>52.0</td>
      <td>19.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows � 22 columns</p>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[62]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre><span></span><span class="c1">#confusion_matrix</span>
<span class="n">cnf_matrix</span> <span class="o">=</span> <span class="n">confusion_matrix</span><span class="p">(</span><span class="n">test</span><span class="p">[</span><span class="s2">&quot;trend&quot;</span><span class="p">],</span> <span class="n">test</span><span class="p">[</span><span class="s2">&quot;predict&quot;</span><span class="p">])</span>
<span class="n">np</span><span class="o">.</span><span class="n">set_printoptions</span><span class="p">(</span><span class="n">precision</span><span class="o">=</span><span class="mi">2</span><span class="p">)</span>
<span class="n">class_names</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;1&#39;</span><span class="p">,</span><span class="s1">&#39;0&#39;</span><span class="p">]</span>

<span class="k">def</span> <span class="nf">plot_confusion_matrix</span><span class="p">(</span><span class="n">cm</span><span class="p">,</span> <span class="n">classes</span><span class="p">,</span>
                          <span class="n">normalize</span><span class="o">=</span><span class="bp">False</span><span class="p">,</span>
                          <span class="n">title</span><span class="o">=</span><span class="s1">&#39;Confusion matrix&#39;</span><span class="p">,</span>
                          <span class="n">cmap</span><span class="o">=</span><span class="n">plt</span><span class="o">.</span><span class="n">cm</span><span class="o">.</span><span class="n">Blues</span><span class="p">):</span>
    <span class="sd">&quot;&quot;&quot;</span>
<span class="sd">    This function prints and plots the confusion matrix.</span>
<span class="sd">    Normalization can be applied by setting `normalize=True`.</span>
<span class="sd">    &quot;&quot;&quot;</span>
    <span class="k">if</span> <span class="n">normalize</span><span class="p">:</span>
        <span class="n">cm</span> <span class="o">=</span> <span class="n">cm</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="s1">&#39;float&#39;</span><span class="p">)</span> <span class="o">/</span> <span class="n">cm</span><span class="o">.</span><span class="n">sum</span><span class="p">(</span><span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)[:,</span> <span class="n">np</span><span class="o">.</span><span class="n">newaxis</span><span class="p">]</span>
        <span class="k">print</span><span class="p">(</span><span class="s2">&quot;Normalized confusion matrix&quot;</span><span class="p">)</span>
    <span class="k">else</span><span class="p">:</span>
        <span class="k">print</span><span class="p">(</span><span class="s1">&#39;Confusion matrix&#39;</span><span class="p">)</span>

    <span class="k">print</span><span class="p">(</span><span class="n">cm</span><span class="p">)</span>

    <span class="n">plt</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">cm</span><span class="p">,</span> <span class="n">interpolation</span><span class="o">=</span><span class="s1">&#39;nearest&#39;</span><span class="p">,</span> <span class="n">cmap</span><span class="o">=</span><span class="n">cmap</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="n">title</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">colorbar</span><span class="p">()</span>
    <span class="n">tick_marks</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">arange</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">classes</span><span class="p">))</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">xticks</span><span class="p">(</span><span class="n">tick_marks</span><span class="p">,</span> <span class="n">classes</span><span class="p">,</span> <span class="n">rotation</span><span class="o">=</span><span class="mi">45</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">yticks</span><span class="p">(</span><span class="n">tick_marks</span><span class="p">,</span> <span class="n">classes</span><span class="p">)</span>

    <span class="n">fmt</span> <span class="o">=</span> <span class="s1">&#39;.2f&#39;</span> <span class="k">if</span> <span class="n">normalize</span> <span class="k">else</span> <span class="s1">&#39;d&#39;</span>
    <span class="n">thresh</span> <span class="o">=</span> <span class="n">cm</span><span class="o">.</span><span class="n">max</span><span class="p">()</span> <span class="o">/</span> <span class="mf">2.</span>
    <span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="n">j</span> <span class="ow">in</span> <span class="n">itertools</span><span class="o">.</span><span class="n">product</span><span class="p">(</span><span class="nb">range</span><span class="p">(</span><span class="n">cm</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">]),</span> <span class="nb">range</span><span class="p">(</span><span class="n">cm</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">1</span><span class="p">])):</span>
        <span class="n">plt</span><span class="o">.</span><span class="n">text</span><span class="p">(</span><span class="n">j</span><span class="p">,</span> <span class="n">i</span><span class="p">,</span> <span class="n">format</span><span class="p">(</span><span class="n">cm</span><span class="p">[</span><span class="n">i</span><span class="p">,</span> <span class="n">j</span><span class="p">],</span> <span class="n">fmt</span><span class="p">),</span>
                 <span class="n">horizontalalignment</span><span class="o">=</span><span class="s2">&quot;center&quot;</span><span class="p">,</span>
                 <span class="n">color</span><span class="o">=</span><span class="s2">&quot;white&quot;</span> <span class="k">if</span> <span class="n">cm</span><span class="p">[</span><span class="n">i</span><span class="p">,</span> <span class="n">j</span><span class="p">]</span> <span class="o">&gt;</span> <span class="n">thresh</span> <span class="k">else</span> <span class="s2">&quot;black&quot;</span><span class="p">)</span>

    <span class="n">plt</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">&#39;True label&#39;</span><span class="p">)</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s1">&#39;Predicted label&#39;</span><span class="p">)</span>
<span class="c1"># Plot non-normalized confusion matrix</span>
<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">()</span>
<span class="n">plot_confusion_matrix</span><span class="p">(</span><span class="n">cnf_matrix</span><span class="p">,</span> <span class="n">classes</span><span class="o">=</span><span class="n">class_names</span><span class="p">,</span>
                      <span class="n">title</span><span class="o">=</span><span class="s1">&#39;Confusion matrix&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>

<span class="k">print</span><span class="p">(</span><span class="n">accuracy_score</span><span class="p">(</span><span class="n">test</span><span class="p">[</span><span class="s2">&quot;trend&quot;</span><span class="p">],</span> <span class="n">test</span><span class="p">[</span><span class="s2">&quot;predict&quot;</span><span class="p">]))</span>
<span class="k">print</span><span class="p">(</span><span class="n">confusion_matrix</span><span class="p">(</span><span class="n">test</span><span class="p">[</span><span class="s2">&quot;trend&quot;</span><span class="p">],</span> <span class="n">test</span><span class="p">[</span><span class="s2">&quot;predict&quot;</span><span class="p">]))</span>
<span class="k">print</span><span class="p">(</span><span class="n">classification_report</span><span class="p">(</span><span class="n">test</span><span class="p">[</span><span class="s2">&quot;trend&quot;</span><span class="p">],</span> <span class="n">test</span><span class="p">[</span><span class="s2">&quot;predict&quot;</span><span class="p">]))</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>Confusion matrix
[[116 162]
 [113 205]]
</pre>
</div>
</div>

<div class="output_area"><div class="prompt"></div>


<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
AABJRU5ErkJggg==
"
>
</div>

</div>

<div class="output_area"><div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>0.538590604027
[[116 162]
 [113 205]]
             precision    recall  f1-score   support

          0       0.51      0.42      0.46       278
          1       0.56      0.64      0.60       318

avg / total       0.53      0.54      0.53       596

</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[63]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre><span></span><span class="k">print</span><span class="p">(</span><span class="n">accuracy_score</span><span class="p">(</span><span class="n">test</span><span class="p">[</span><span class="s2">&quot;trend&quot;</span><span class="p">],</span> <span class="n">test</span><span class="p">[</span><span class="s2">&quot;predict&quot;</span><span class="p">]))</span>
<span class="k">print</span><span class="p">(</span><span class="n">confusion_matrix</span><span class="p">(</span><span class="n">test</span><span class="p">[</span><span class="s2">&quot;trend&quot;</span><span class="p">],</span> <span class="n">test</span><span class="p">[</span><span class="s2">&quot;predict&quot;</span><span class="p">]))</span>
<span class="k">print</span><span class="p">(</span><span class="n">classification_report</span><span class="p">(</span><span class="n">test</span><span class="p">[</span><span class="s2">&quot;trend&quot;</span><span class="p">],</span> <span class="n">test</span><span class="p">[</span><span class="s2">&quot;predict&quot;</span><span class="p">]))</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>0.538590604027
[[116 162]
 [113 205]]
             precision    recall  f1-score   support

          0       0.51      0.42      0.46       278
          1       0.56      0.64      0.60       318

avg / total       0.53      0.54      0.53       596

</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="AdaBoost-Classifier">AdaBoost Classifier<a class="anchor-link" href="#AdaBoost-Classifier">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[64]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre><span></span><span class="c1">#clf = GradientBoostingClassifier(n_estimators=1000, max_depth=10)</span>
<span class="n">clf</span> <span class="o">=</span> <span class="n">AdaBoostClassifier</span><span class="p">(</span><span class="n">n_estimators</span><span class="o">=</span><span class="mi">100</span><span class="p">)</span>
<span class="c1">#clf = linear_model.LinearRegression()</span>

<span class="n">clf</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X</span><span class="p">,</span><span class="n">y</span><span class="p">)</span>
<span class="n">test</span><span class="p">[</span><span class="s2">&quot;predict&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">clf</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">test_X</span><span class="p">)</span>
<span class="n">test</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">5</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[64]:</div>

<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PosCount</th>
      <th>NegCount</th>
      <th>TrustCount</th>
      <th>AngerCount</th>
      <th>AnticipationCount</th>
      <th>DisgustCount</th>
      <th>FearCount</th>
      <th>JoyCount</th>
      <th>SadnessCount</th>
      <th>SurpriseCount</th>
      <th>...</th>
      <th>TrustCount_cv</th>
      <th>AngerCount_cv</th>
      <th>AnticipationCount_cv</th>
      <th>DisgustCount_cv</th>
      <th>FearCount_cv</th>
      <th>JoyCount_cv</th>
      <th>SadnessCount_cv</th>
      <th>SurpriseCount_cv</th>
      <th>trend</th>
      <th>predict</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1393</th>
      <td>30</td>
      <td>35</td>
      <td>27</td>
      <td>20</td>
      <td>9</td>
      <td>10</td>
      <td>31</td>
      <td>6</td>
      <td>17</td>
      <td>9</td>
      <td>...</td>
      <td>99.0</td>
      <td>82.0</td>
      <td>46.0</td>
      <td>44.0</td>
      <td>125.0</td>
      <td>27.0</td>
      <td>70.0</td>
      <td>29.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1394</th>
      <td>25</td>
      <td>33</td>
      <td>20</td>
      <td>20</td>
      <td>12</td>
      <td>5</td>
      <td>29</td>
      <td>10</td>
      <td>14</td>
      <td>3</td>
      <td>...</td>
      <td>107.0</td>
      <td>77.0</td>
      <td>54.0</td>
      <td>38.0</td>
      <td>112.0</td>
      <td>41.0</td>
      <td>63.0</td>
      <td>30.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1395</th>
      <td>23</td>
      <td>25</td>
      <td>23</td>
      <td>13</td>
      <td>12</td>
      <td>5</td>
      <td>11</td>
      <td>10</td>
      <td>10</td>
      <td>7</td>
      <td>...</td>
      <td>109.0</td>
      <td>76.0</td>
      <td>57.0</td>
      <td>36.0</td>
      <td>98.0</td>
      <td>42.0</td>
      <td>59.0</td>
      <td>30.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1396</th>
      <td>23</td>
      <td>22</td>
      <td>10</td>
      <td>9</td>
      <td>5</td>
      <td>6</td>
      <td>15</td>
      <td>4</td>
      <td>11</td>
      <td>2</td>
      <td>...</td>
      <td>92.0</td>
      <td>65.0</td>
      <td>53.0</td>
      <td>32.0</td>
      <td>82.0</td>
      <td>40.0</td>
      <td>53.0</td>
      <td>23.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1397</th>
      <td>27</td>
      <td>22</td>
      <td>19</td>
      <td>13</td>
      <td>11</td>
      <td>3</td>
      <td>18</td>
      <td>8</td>
      <td>9</td>
      <td>4</td>
      <td>...</td>
      <td>85.0</td>
      <td>68.0</td>
      <td>47.0</td>
      <td>27.0</td>
      <td>88.0</td>
      <td>36.0</td>
      <td>52.0</td>
      <td>19.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows � 22 columns</p>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[65]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre><span></span><span class="k">print</span><span class="p">(</span><span class="n">accuracy_score</span><span class="p">(</span><span class="n">test</span><span class="p">[</span><span class="s2">&quot;trend&quot;</span><span class="p">],</span> <span class="n">test</span><span class="p">[</span><span class="s2">&quot;predict&quot;</span><span class="p">]))</span>
<span class="k">print</span><span class="p">(</span><span class="n">confusion_matrix</span><span class="p">(</span><span class="n">test</span><span class="p">[</span><span class="s2">&quot;trend&quot;</span><span class="p">],</span> <span class="n">test</span><span class="p">[</span><span class="s2">&quot;predict&quot;</span><span class="p">]))</span>
<span class="k">print</span><span class="p">(</span><span class="n">classification_report</span><span class="p">(</span><span class="n">test</span><span class="p">[</span><span class="s2">&quot;trend&quot;</span><span class="p">],</span> <span class="n">test</span><span class="p">[</span><span class="s2">&quot;predict&quot;</span><span class="p">]))</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>0.468120805369
[[ 97 181]
 [136 182]]
             precision    recall  f1-score   support

          0       0.42      0.35      0.38       278
          1       0.50      0.57      0.53       318

avg / total       0.46      0.47      0.46       596

</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Neural-Network">Neural Network<a class="anchor-link" href="#Neural-Network">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[82]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre><span></span><span class="c1">#clf = GradientBoostingClassifier(n_estimators=1000, max_depth=10)</span>
<span class="c1">#clf = AdaBoostClassifier(n_estimators=100)</span>
<span class="c1">#clf = linear_model.LinearRegression()</span>
<span class="kn">from</span> <span class="nn">sklearn.neural_network</span> <span class="kn">import</span> <span class="n">MLPClassifier</span>
<span class="n">clf</span> <span class="o">=</span> <span class="n">MLPClassifier</span><span class="p">(</span><span class="n">solver</span><span class="o">=</span><span class="s1">&#39;lbfgs&#39;</span><span class="p">,</span> <span class="n">alpha</span><span class="o">=</span><span class="mf">1e-5</span><span class="p">,</span> <span class="n">hidden_layer_sizes</span><span class="o">=</span><span class="p">(</span><span class="mi">10</span><span class="p">,</span><span class="mi">10</span><span class="p">),</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>

<span class="n">clf</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X</span><span class="p">,</span><span class="n">y</span><span class="p">)</span>
<span class="n">test</span><span class="p">[</span><span class="s2">&quot;predict&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">clf</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">test_X</span><span class="p">)</span>
<span class="n">test</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">5</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[82]:</div>

<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PosCount</th>
      <th>NegCount</th>
      <th>TrustCount</th>
      <th>AngerCount</th>
      <th>AnticipationCount</th>
      <th>DisgustCount</th>
      <th>FearCount</th>
      <th>JoyCount</th>
      <th>SadnessCount</th>
      <th>SurpriseCount</th>
      <th>...</th>
      <th>TrustCount_cv</th>
      <th>AngerCount_cv</th>
      <th>AnticipationCount_cv</th>
      <th>DisgustCount_cv</th>
      <th>FearCount_cv</th>
      <th>JoyCount_cv</th>
      <th>SadnessCount_cv</th>
      <th>SurpriseCount_cv</th>
      <th>trend</th>
      <th>predict</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1393</th>
      <td>30</td>
      <td>35</td>
      <td>27</td>
      <td>20</td>
      <td>9</td>
      <td>10</td>
      <td>31</td>
      <td>6</td>
      <td>17</td>
      <td>9</td>
      <td>...</td>
      <td>99.0</td>
      <td>82.0</td>
      <td>46.0</td>
      <td>44.0</td>
      <td>125.0</td>
      <td>27.0</td>
      <td>70.0</td>
      <td>29.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1394</th>
      <td>25</td>
      <td>33</td>
      <td>20</td>
      <td>20</td>
      <td>12</td>
      <td>5</td>
      <td>29</td>
      <td>10</td>
      <td>14</td>
      <td>3</td>
      <td>...</td>
      <td>107.0</td>
      <td>77.0</td>
      <td>54.0</td>
      <td>38.0</td>
      <td>112.0</td>
      <td>41.0</td>
      <td>63.0</td>
      <td>30.0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1395</th>
      <td>23</td>
      <td>25</td>
      <td>23</td>
      <td>13</td>
      <td>12</td>
      <td>5</td>
      <td>11</td>
      <td>10</td>
      <td>10</td>
      <td>7</td>
      <td>...</td>
      <td>109.0</td>
      <td>76.0</td>
      <td>57.0</td>
      <td>36.0</td>
      <td>98.0</td>
      <td>42.0</td>
      <td>59.0</td>
      <td>30.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1396</th>
      <td>23</td>
      <td>22</td>
      <td>10</td>
      <td>9</td>
      <td>5</td>
      <td>6</td>
      <td>15</td>
      <td>4</td>
      <td>11</td>
      <td>2</td>
      <td>...</td>
      <td>92.0</td>
      <td>65.0</td>
      <td>53.0</td>
      <td>32.0</td>
      <td>82.0</td>
      <td>40.0</td>
      <td>53.0</td>
      <td>23.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1397</th>
      <td>27</td>
      <td>22</td>
      <td>19</td>
      <td>13</td>
      <td>11</td>
      <td>3</td>
      <td>18</td>
      <td>8</td>
      <td>9</td>
      <td>4</td>
      <td>...</td>
      <td>85.0</td>
      <td>68.0</td>
      <td>47.0</td>
      <td>27.0</td>
      <td>88.0</td>
      <td>36.0</td>
      <td>52.0</td>
      <td>19.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows � 22 columns</p>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[83]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre><span></span><span class="k">print</span><span class="p">(</span><span class="n">accuracy_score</span><span class="p">(</span><span class="n">test</span><span class="p">[</span><span class="s2">&quot;trend&quot;</span><span class="p">],</span> <span class="n">test</span><span class="p">[</span><span class="s2">&quot;predict&quot;</span><span class="p">]))</span>
<span class="k">print</span><span class="p">(</span><span class="n">confusion_matrix</span><span class="p">(</span><span class="n">test</span><span class="p">[</span><span class="s2">&quot;trend&quot;</span><span class="p">],</span> <span class="n">test</span><span class="p">[</span><span class="s2">&quot;predict&quot;</span><span class="p">]))</span>
<span class="k">print</span><span class="p">(</span><span class="n">classification_report</span><span class="p">(</span><span class="n">test</span><span class="p">[</span><span class="s2">&quot;trend&quot;</span><span class="p">],</span> <span class="n">test</span><span class="p">[</span><span class="s2">&quot;predict&quot;</span><span class="p">]))</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>0.541946308725
[[ 38 240]
 [ 33 285]]
             precision    recall  f1-score   support

          0       0.54      0.14      0.22       278
          1       0.54      0.90      0.68       318

avg / total       0.54      0.54      0.46       596

</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Naive-Bayes">Naive Bayes<a class="anchor-link" href="#Naive-Bayes">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[70]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre><span></span><span class="c1"># Naive Bayes Algo</span>
<span class="kn">from</span> <span class="nn">sklearn.naive_bayes</span> <span class="kn">import</span> <span class="n">GaussianNB</span>
<span class="n">clf</span> <span class="o">=</span> <span class="n">GaussianNB</span><span class="p">()</span>
<span class="n">clf</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">)</span>
<span class="n">test</span><span class="p">[</span><span class="s2">&quot;predict&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">clf</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">test_X</span><span class="p">)</span>
<span class="n">test</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">5</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt output_prompt">Out[70]:</div>

<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PosCount</th>
      <th>NegCount</th>
      <th>TrustCount</th>
      <th>AngerCount</th>
      <th>AnticipationCount</th>
      <th>DisgustCount</th>
      <th>FearCount</th>
      <th>JoyCount</th>
      <th>SadnessCount</th>
      <th>SurpriseCount</th>
      <th>...</th>
      <th>TrustCount_cv</th>
      <th>AngerCount_cv</th>
      <th>AnticipationCount_cv</th>
      <th>DisgustCount_cv</th>
      <th>FearCount_cv</th>
      <th>JoyCount_cv</th>
      <th>SadnessCount_cv</th>
      <th>SurpriseCount_cv</th>
      <th>trend</th>
      <th>predict</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1393</th>
      <td>30</td>
      <td>35</td>
      <td>27</td>
      <td>20</td>
      <td>9</td>
      <td>10</td>
      <td>31</td>
      <td>6</td>
      <td>17</td>
      <td>9</td>
      <td>...</td>
      <td>99.0</td>
      <td>82.0</td>
      <td>46.0</td>
      <td>44.0</td>
      <td>125.0</td>
      <td>27.0</td>
      <td>70.0</td>
      <td>29.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1394</th>
      <td>25</td>
      <td>33</td>
      <td>20</td>
      <td>20</td>
      <td>12</td>
      <td>5</td>
      <td>29</td>
      <td>10</td>
      <td>14</td>
      <td>3</td>
      <td>...</td>
      <td>107.0</td>
      <td>77.0</td>
      <td>54.0</td>
      <td>38.0</td>
      <td>112.0</td>
      <td>41.0</td>
      <td>63.0</td>
      <td>30.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1395</th>
      <td>23</td>
      <td>25</td>
      <td>23</td>
      <td>13</td>
      <td>12</td>
      <td>5</td>
      <td>11</td>
      <td>10</td>
      <td>10</td>
      <td>7</td>
      <td>...</td>
      <td>109.0</td>
      <td>76.0</td>
      <td>57.0</td>
      <td>36.0</td>
      <td>98.0</td>
      <td>42.0</td>
      <td>59.0</td>
      <td>30.0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1396</th>
      <td>23</td>
      <td>22</td>
      <td>10</td>
      <td>9</td>
      <td>5</td>
      <td>6</td>
      <td>15</td>
      <td>4</td>
      <td>11</td>
      <td>2</td>
      <td>...</td>
      <td>92.0</td>
      <td>65.0</td>
      <td>53.0</td>
      <td>32.0</td>
      <td>82.0</td>
      <td>40.0</td>
      <td>53.0</td>
      <td>23.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1397</th>
      <td>27</td>
      <td>22</td>
      <td>19</td>
      <td>13</td>
      <td>11</td>
      <td>3</td>
      <td>18</td>
      <td>8</td>
      <td>9</td>
      <td>4</td>
      <td>...</td>
      <td>85.0</td>
      <td>68.0</td>
      <td>47.0</td>
      <td>27.0</td>
      <td>88.0</td>
      <td>36.0</td>
      <td>52.0</td>
      <td>19.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows � 22 columns</p>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[71]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre><span></span><span class="k">print</span><span class="p">(</span><span class="n">accuracy_score</span><span class="p">(</span><span class="n">test</span><span class="p">[</span><span class="s2">&quot;trend&quot;</span><span class="p">],</span> <span class="n">test</span><span class="p">[</span><span class="s2">&quot;predict&quot;</span><span class="p">]))</span>
<span class="k">print</span><span class="p">(</span><span class="n">confusion_matrix</span><span class="p">(</span><span class="n">test</span><span class="p">[</span><span class="s2">&quot;trend&quot;</span><span class="p">],</span> <span class="n">test</span><span class="p">[</span><span class="s2">&quot;predict&quot;</span><span class="p">]))</span>
<span class="k">print</span><span class="p">(</span><span class="n">classification_report</span><span class="p">(</span><span class="n">test</span><span class="p">[</span><span class="s2">&quot;trend&quot;</span><span class="p">],</span> <span class="n">test</span><span class="p">[</span><span class="s2">&quot;predict&quot;</span><span class="p">]))</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area"><div class="prompt"></div>
<div class="output_subarea output_stream output_stdout output_text">
<pre>0.506711409396
[[ 78 200]
 [ 94 224]]
             precision    recall  f1-score   support

          0       0.45      0.28      0.35       278
          1       0.53      0.70      0.60       318

avg / total       0.49      0.51      0.48       596

</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered">
<div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Conclusion:">Conclusion:<a class="anchor-link" href="#Conclusion:">&#182;</a></h3><p>The maximum accuracy of predicting overall stock market trend is 55%. It implies that sentiment of news headlines have the impact on stock market trend.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython2"><pre><span></span> 
</pre></div>

</div>
</div>
</div>

</div>
    </div>
  </div>
</body>
</html>
