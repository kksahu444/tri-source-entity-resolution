# Error analysis: where macro F0.5 is lost (dev OOF, rule `gfm_indep`)


## Loss decomposition (share of 1 - macro F0.5)

| Component | Loss | Share |
|---|---|---|
| false links | 0.00453 | 23.2% |
| model misses | 0.00900 | 46.1% |
| blocking | 0.00598 | 30.7% |

## By slice (mean loss per entity x share of entities = contribution)

| Slice | Entities | FP loss | Model-miss loss | Blocking loss |
|---|---|---|---|---|
| all | 100.0% | 0.00453 | 0.00900 | 0.00598 |
| US | 60.7% | 0.00233 | 0.00528 | 0.00265 |
| India | 39.3% | 0.00220 | 0.00372 | 0.00333 |
| true singletons | 5.6% | 0.00143 | 0.00000 | 0.00000 |
| 1 true match | 5.4% | 0.00057 | 0.00170 | 0.00098 |
| 2-3 true matches | 41.1% | 0.00163 | 0.00372 | 0.00254 |
| 4+ true matches | 47.9% | 0.00090 | 0.00358 | 0.00246 |
| name-city crowd >= 3 | 3.3% | 0.00014 | 0.00036 | 0.00083 |
| has a blocked non-Latin record | 0.4% | 0.00001 | 0.00007 | 0.00072 |
| has a blocked empty-address record | 2.1% | 0.00005 | 0.00019 | 0.00194 |

### India: false links (selected, not a match)

- `Gangan Solutions Associates | N.H.31, Near Gayan Bharti School Po - Begusarai, Ps - Begusarai, Begusarai, Bihar`  vs  `Ganngan Solutions Associates | N.H.31/4, NEAR GAYAN BHARTI SCHOOL PO - BEGUSARAI, PS - BEGUSARAI, बिहार`
- `Square (India) Apparel Private Limited | 108, Anant Vihar Colony Iim Road, Near Mubarakpur, Lucknow, Uttar Pradesh`  vs  `square (india) apparel private limited | `
- `Devanagere District Retail Limited | Karnataka, H.No.153, Devanagere, Devanagere District, Huchavvanahalli Devanagere Taluk`  vs  `Devanagere District Marketing Limited Services | Karnataka, H.NO.153, HUCHAVVANAHALLI DEVANAGERE TALUK, DEVANAGERE DISTRICT`
- `Sma Tech Private Limited | Ap Square, No. 302, 2Nd Floor, 5Th D Main 2Nd Block, 3Rd Cross, Hrbr Layout, Kalyan N, Bangalore, Karnataka`  vs  `Limited Sma Carriers Private | BANGALORE, H.NO 19 AP SQUARE, BENGALURU, NO. 302, 2ND FLOOR, 5TH D MAIN 2ND BLOCK, 3RD CROSS, HRBR LAYOUT, KALYAN N, Karnataka`
- `Anand Foundation Private Limited | Plot No-B/54, Sector-10, Cuttack Sadar, Cuttack, Orissa`  vs  `ଆନନ୍ଦ ଫାଉଣ୍ଡେସନ୍ ପ୍ରାଇଭେଟ୍ ଲିମିଟେଡ୍ | OFFICE OF SHRI BIJAN RAYSENIOR ADVOCATE SUNSHINE FIELD EAST, CUTTACK, Odisha`
- `Juno & Co | C/9 Industrial Estate Chandaka, Bhubaneswar, Khordha, Orissa`  vs  `@juno | `
- `Girraj Pack Ltd | P 482/3 Sunit Mahaganpati Rees Khalapur-Raigad Rees, Panvel, Raigarh, Maharashtra`  vs  `Girraj Pack Pvt Ltd | P 482/3- SUNIT MAHAGANPATI REES KHALAPUR-RAIGAD REES, PANVEL, Maharashtra`
- `Manav Paper Private Limited | Aurangabad, Maharashtra, Ulkanagri Garkheda, Opp.Sant Eknath Hospital D-43 Shrinagar, Aurangabad`  vs  `Public Manav Paper Limited | OPP.SANT EKNATH HOSPITAL D-54 SHRINAGAR, ULKANAGRI GARKHEDA, AURANGABAD, Maharashtra`
- `Westend Trading Private Limited | Near Shivam Estate, Chhatral Gidc, Gandhi Nagar, 1719, Gujarat, Kalol`  vs  `Private Westend Limited Overseas Trading | NEAR SHIVAM ESTATE, CHHATRAL GIDC, GANDHI NAGAR, Gujarat`
- `CS Developers Limited | Plot No X 24 Gidc Electronic Estate Sector 25, Gandhinagar, Gandhi Nagar, Gujarat`  vs  `Srv Services | Plot No.x-24, Gidc Electronic Estate, Sector 25, Gandhinagar, Gandhi Nagar, Gandhinagar, GJ`
- `Black Foods Private Limited | M-36, Guru Harkishan Nagar Paschim Vihar, New Delhi, West Delhi, Delhi`  vs  `Black Foods | B-36, East Delhi, Delhi, दिल्ली`
- `Bangalore Services Private Limited | Karnataka, 239 No 16 I Floor Sns Chambers Sankey Road Sadashivanagar, Bangalore`  vs  `servicesbangalore.com | H.no 239 No 16 I Floor Sns Chambers Sankey Road Sadashivanagar, Bangalore, KA`

### India: true matches retrieved but not selected

- `Balaji Solutions Private Limited | 18-H, Shayam Sadan, Opposite Sahara Appartment Daudpur, Gorakhpur, Uttar Pradesh`  vs  `Balaji Prívate Limited Service | 8-H, Gorakhpur, उत्तर प्रदेश`
- `Gold Infotech Private Limited | Marrot, Jahajgarh, Jhajjar, Haryana`  vs  `Zetasolcira | Rohtak, Marrot, Jhajjar, हरियाणा`
- `Savera Technologies Ltd | No, 81, 2Nd Main Road, Subbanna Garden, Bangalore North, Bangalore, Karnataka`  vs  `Savera Technologies [Ltd] | `
- `Ganaur Construction Pvt. Ltd. | Ward No. 3, Hari Nagar, Near Janta School, Ganaur, Sonipat, Haryana`  vs  `Ganaur Pvt. Ltd. Center | `
- `Mars & Co | Santhi Bhavan, Kra-40(4/472) Kanchiyoorkonam, Kattakada P.O, Thiruvananthapuram, Trivandrum, Kerala`  vs  `Mars & [Co] | `
- `Properties City Chemicals Private Limited | 385, Ground Floor, Lig Flats Pocket-13, Phase-1, Manglapuri, Dwarka, New Delhi, West Delhi, Delhi`  vs  `Properties Cdty Chemicals-Private Limited | 385, GROUND FLOOR, LIG FLATS POCKET-13, PHASE-, MANGLAPURI, DWARKA, NEW DELHI, Delhi`
- `Neel Buildcon Private Limited | Krishna, Vijayawada (Urban), Andhra Pradesh, 608 Sri Ramachandra Ngr, Reliance Island Garden`  vs  `Dr Neel Blucdn Private Limited | Block F-988 Reliance Island Garden, Vijayawada (Urban), Krishna, AP`
- `Jyoti Developers Limited | Sanjib Sarani Durgapurburdwan, Burdwan, Medinipur, West Bengal`  vs  `Jyoti Developers Limited | `
- `Bangalore Pharmaceutical Private Limited | No 32, 7Th Cross, Manjunathnagar, Magadi Road, Bangalore, Karnataka`  vs  `BANGALORE PHARMACEUTICAL | `
- `Silver Investment Private Limited | Ground Floor, House No 40/13, New Delhi, Central Delhi, Delhi`  vs  `M/s Silver Investment Private Limited | `
- `Galaxy Investments LLP | D. No. 13-8-197/C, Yamini Plaza Opp. Rtc Bus Stand, Guntur, Andhra Pradesh`  vs  `Galaxy LLP Investments | `
- `Aditya Exports Pvt Ltd | Flat No-311, 3Rd Floor Block-B Brindavanam Apt Panthur Main Road, Kadubisnahalli, Bangalore, Karnataka`  vs  `Aditya-Exports Pvt | `

### India: true matches never retrieved

- `Dynamic Estate Private Limited | Pvt Fno-1, Delhi, West Delhi, Delhi, Ugf Plot-234A/235/A-1Kh No-154 Vipin Garden Uttam Nagar Nr Som Bazar Rd`  vs  `Onyxonyxumbra | Pvt Fno-1, Delhi, West Delhi, DL`
- `New Delhi Trading Private Limited | N-107, 3Rd Floor, Greater Kailash-I, New Delhi, South Delhi, Delhi`  vs  `New Delhi Trafncg Private Limited | `
- `Smart Indian Foods | 94/06, Model House Street, Basavangudi, Bangalore, Karnataka`  vs  `Smart Icndnha Foods | `
- `Sun Infra Private Limited | 4384/4A, Gobind Bhavan, Ansari Road Darya Ganj, New Delhi, Central Delhi, Delhi`  vs  `siprivate.com | 4A, New Delhi, Central Delhi, DL`
- `Arvind Institute of Technology Private Limited | No.3/1, Sri Narayan Mansion, Deewan Madhava Rao Road, Opp Mnk Rao Park, Bangalore South, Bangalore, Karnataka`  vs  `Miraxylodelta | NO.3/1, NULL, BENGALURU, BANGALORE, Karnataka`
- `Vistaar Traders Private Limited | Gf-38, Indra Prakash Building, 21, Barakhamba Road, New Delhi, Central Delhi, Delhi`  vs  `Vistaar Vistaar Tratteis Private [Limited] | `
- `Apex Power Private Limited | No. 51, Old Site No. 1, Business Synergy Centre, Fifth Floor, 5Th Main, 5Th Block, Bangalore South, Bangalore, Karnataka`  vs  `Apex Power Limited Service | 051, Bengaluru, Bangalore, KA`
- `Saraswati & Associates | No 393, Cj Venkata Das Road, Bangalore South, Bangalore, Karnataka`  vs  `(+) ASSOCIATES SARASWATI | H.NO 6-393, BANGALORE SOUTH, BANGALORE, Karnataka`
- `Srija Brothers Private Limited | Room No.20, 4Th Floor, Sai Suyog Building, Sai Nagar, Plot No.482, 52-Bunglow Area, Panvel, Maharashtra`  vs  `Verapyra | Room No.20, Panvel, महाराष्ट्र`
- `Virtual (India) Consultants Private Limited | Madhya Pradesh, Indore, 504 Nrk Business Park Behind C21 Mall, Pu-4, Vijay Nagar, Indore, Office No. 6`  vs  `M/s Virtual (India) Consultants Private | Office No. 06, Indore, मध्य प्रदेश`
- `Lord Hospital Private Limited | #396, 2Nd Cross, 2Nd Main, Rmv 2Nd Stage Dollars Colony, Sanjay Nagar, Bangalore, Karnataka`  vs  `Lord Hsidstal Private Limited | Deor No 396, Bangalore, Ele Kodigehalli, KA`
- `Southern Care Private Limited | 12/2 Main Road, Sec-37, Faridabad, Faridabad, Haryana`  vs  `Southern Care Private Enterprises | `

### US: false links (selected, not a match)

- `Hatfield & Yohannes Neuberger LLC | 2020 Jewell Lane, CA, Redding`  vs  `Hatfield LLC Yohannes Neuberger & | `
- `Capital Fellowship IV | 303 Eagle Crest Drive, Poteau, OK`  vs  `Capital Fellowship EV | 303-307 EAGLE CREST DRIVE, POTEAU, OK`
- `TN Produce LLC | 6902 Rothwell Street, Bldg 12, New Albany, OH`  vs  `FN Produce LLC | 6902 Rothwell St, Bldg. 12, NEW Albany CITY, Ohio`
- `2424 Union Blvd Owners Corp, Inc | 2021 40th Street, Lubbock, TX`  vs  `2424 Union Bsrdl Owners Corp, Inc | `
- `Litaor Inc | 2703 Bringhurst Street, Houston, TX`  vs  `Fior Inc | 2703 Bringhurst St, Houston, Texas`
- `Diaz and Kirkman Worldwide | 20214 Upper Greatland Drive, Chugiak, AK`  vs  `Huff and Worldwide Kirkman | #20214 UPPER GREATLAND DR, CHUGIAK, AK`
- `Rocky Council III Inc | 17 Carter Court, Allen, TX`  vs  `Inc Rocky Coucli II | 17 Carter Ct, Allen, Texas`
- `Orbon Maintenance | 348 Pine Needle Lane, Unit 407, Burlington, NC`  vs  `ORRBON MAINTEANCE MAINTEANCE | BURLINGTON, 48 PINE NEEDLE LN, NC`
- `Brayman Dental Associates of Fort Mill | 5056 Kettering Avenue, Fort Mill, SC`  vs  `tónysteers.com | FT. MILL, SC, 005056 KETTERING AVE`
- `Curlee's Royal Partners | 832 Luster Avenue, Springfield, MO`  vs  `Curlee's Royal Partners LLC | `
- `Cohen and Price State | 429 Flag Pond Road, Saco, ME`  vs  `Cochran and Price State | 429 Flag Pond Road, Saco, Maine`
- `Southard, Winnah G., M.D., M.D., P.C. | 3421 Willow Wick Trail, Springfield, MO`  vs  `SOUTHARD, WINNAH G., M.D., M.D., P.C. | `

### US: true matches retrieved but not selected

- `Shellenberger Gaming Inc | 20 Murray Street, Babylon, NY`  vs  `Shellenberger Gaming | `
- `Mcdonald, Farrow & Smith L.L.C. | 8030 Carolwood Drive, Houston, TX`  vs  `Mcdonald, Farrow & Smith-L.L.C. | 8010 CAROLWOOD DR, HOUSTON CIITY, TX`
- `Dana Gregory Reliable Realty Group | 121 Brackett Road, Newton, MA`  vs  `Dana Gregory Reliable Realty (ID: 85349) | `
- `Endyne LLC | 17 Van Wort Drive, Haverstraw, NY`  vs  `Endyne-Llc | `
- `Best Wealth Industries LLC | 229 Pearl Street, Unit 2, Malden, MA`  vs  `Onyxdrex | Massachusetts, Pearl St, Malden, # 2`
- `Sanders & Rynders PLLC | 8456 Euclid Avenue, Chicago, IL`  vs  `Sanders & Rynders P.L.L.C. | `
- `Martin Esports | 2157 Oxford Road, Redding, CA`  vs  `Martin (Esports) | `
- `Hale Modern Biotech | 402 Bembridge Lane, Franklin, TN`  vs  `HALE MODERN BIOTECH | `
- `Long Inc. | 208 Brannon Drive, Murray, NE`  vs  `Long Inc. | `
- `Metropolitan Fellowship | 12505 102nd Way, Vancouver, WA`  vs  `METROPOLITAN FELLOWSHIP Enterprises | `
- `Premier Nutrition Works L.L.C. | 106 10th Street, Milford, DE`  vs  `Premier Nutrition-Works | `
- `Howell, Brewer & Goolsby LLC | TX, Rowlett, 8604 Cross Timber Trail`  vs  `Kelofaye | 8604 Cross Timber Trl, Rowlett, Texas`

### US: true matches never retrieved

- `Ace Live LLC | 103 Kenton Court, New Castle, DE`  vs  `Ace LLC Partners | www.acellcpa.com | `
- `Behavioral Health Care Inc. | 10860 Buttercup Place, Unit UNIT 201, Prince William County, VA`  vs  `BEHAVIORAL HEALTH CARE INC. Trading | `
- `Cure Deli! Inc. | 2917 Lewis Avenue, Waukegan, IL`  vs  `CURE DELI! INCORPORATED | `
- `Hernandez Allurion | 13580 Goodview Avenue, Hugo, MN`  vs  `Hernandez-Aluion | `
- `Sosonvue LLC | 3317 L Street, Unit Apt 26, Bentonville, AR`  vs  `Sosbnlue LLC | 3317-3321 L St, # Apt 26, Betnonville, Arkansas`
- `Naa | 16606 W Catawba Avenue, Huntersville, NC`  vs  `Naa-Services | `
- `Commercial Software Technologies LLC | 125 3rd Street, Pacific, MO`  vs  `Ariaarcpyra | 125 Third St, PO Box 9358, Pacific, Missouri`
- `Dental Care Associates P.C. | Lubbock, 5004 70th Street, TX`  vs  `Dental Care Associates Enterprises | `
- `American Charities III | 24 Belcher Street, Sharon, MA`  vs  `AMERICAN CHARITIES | BELCHER ST, MA, SHARRON`
- `Jones Rent L.L.C. | 111 Stuart Place, Washington, NC`  vs  `Tavonex | WASHINTGON, 122 STUART PLACE, NC`
- `Verrance Hyperliquid LLC | 10864 Arrington Drive, Rancho Cordova, CA`  vs  `Belonylagild | 11092 ARRINGTON DR, RANCHO CORDOVA, CA`
- `24 Hour Towing | Cedar Park, Unit 901, 1701 Bell Boulevard, TX`  vs  `24 Hour Enterprises | `
