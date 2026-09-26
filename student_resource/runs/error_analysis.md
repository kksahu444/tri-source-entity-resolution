# Error analysis: where macro F0.5 is lost (dev OOF, rule `gfm_indep`)


## Loss decomposition (share of 1 - macro F0.5)

| Component | Loss | Share |
|---|---|---|
| false links | 0.00469 | 22.3% |
| model misses | 0.00683 | 32.5% |
| blocking | 0.00950 | 45.2% |

## By slice (mean loss per entity x share of entities = contribution)

| Slice | Entities | FP loss | Model-miss loss | Blocking loss |
|---|---|---|---|---|
| all | 100.0% | 0.00469 | 0.00683 | 0.00950 |
| US | 60.7% | 0.00234 | 0.00394 | 0.00484 |
| India | 39.3% | 0.00235 | 0.00289 | 0.00466 |
| true singletons | 5.6% | 0.00152 | 0.00000 | 0.00000 |
| 1 true match | 5.4% | 0.00056 | 0.00129 | 0.00162 |
| 2-3 true matches | 41.1% | 0.00167 | 0.00288 | 0.00402 |
| 4+ true matches | 47.9% | 0.00094 | 0.00265 | 0.00386 |
| name-city crowd >= 3 | 3.3% | 0.00015 | 0.00029 | 0.00096 |
| has a blocked non-Latin record | 0.4% | 0.00001 | 0.00005 | 0.00077 |
| has a blocked empty-address record | 5.9% | 0.00016 | 0.00046 | 0.00557 |

### India: false links (selected, not a match)

- `Gama Memorial Trust LLP | Off No.5 D-211 Iind Floor Laxman Singh Complex-Iii Village Munirka, Delhi, South Delhi, Delhi`  vs  `Gama Memorial Trust | `
- `Tejtech Chemical Pvt Ltd | Ist Floor Village Hasangarh, Tehsil Gohana, Gohana, Sonipat, Haryana`  vs  `Tejtechyn Chemical Pvt Ltd | Ist Floor Village Hasangarh, Tehsil Gohana, Gohana, Chidana, HR`
- `New Delhi India Private Limited | T-20 Khanna Marketwest Patel Nagar, New Delhi, West Delhi, Delhi`  vs  `New Delhi India Private Ltd | T-20 Khanna Marketwest Patel Nagar, New Delhi, West Delhi, DL`
- `YU Trading Private Limited | Door No 26/99/3B, Jalils Building, Vythiri, Wayanad, Kerala`  vs  `YU Trading Ltd Ltd | `
- `Southern Royal Energy Private Limited | Gali No. 2 Bhajanpura, New Delhi, C-56/6, Delhi`  vs  `Avinylagild | C-56/6, Gali No. 2 Bhajanpura, New Delhi, Delhi`
- `Lotus Consulting Pvt Ltd | No. 2 Sonnappa Block, 3Rd Cross, Bangalore 46, Bangalore, Karnataka`  vs  `Lotus Consulting Limited | NO. 3 SONNAPPA BLOCK, 3RD CROSS, BANGALORE 46, ಕರ್ನಾಟಕ`
- `East Care Private Limited | 601, Capitol Icon, Opp. Swagat High Street, Gandhinagar, Gandhi Nagar, Gujarat`  vs  `lawchambers.com | Gujarat, CAPITOL ICON-1, OPP. SWAGAT HIGH STREET, GANDHI NAGAR, J-101, GANDHINAGAR`
- `India Jc Awas Private Limited | Plot No - 293, Ga Colony Bharatpur, Bhubaneswar, Khordha, Orissa`  vs  `India Jc Awas Prívate Ltd | `
- `DWB Shelter Private Limited | Flat No 203, Palm Prem Shanti Residency Shreeram Enclave Kumerd Tehsil Sanwer, Indore, Madhya Pradesh`  vs  `ínfrastructuregroup.com | H.NO 03 , PALM PREM SHANTI RESIDENCY SHREERAM ENCLAVE KUMERD TEHSIL SANWER, INDORE, Madhya Pradesh`
- `Dhillon Charitable Trust | Flat No.304, Vijaya Apartment, 1St Line, Svn Colony, Guntur, Andhra Pradesh`  vs  `Dhillon Charitable Trust Cae | `
- `Northeast Services (India) Limited | Building No. 17/4, Thottumpuram Building Nanniyode, Pacha P O, Thiruvananthapuram, Trivandrum, Kerala`  vs  `Sri ... Indian 5ervices | Plot 126 Building No. 17/4, Thottumpuram Building Nanniyode, Pacha P O, Thiruvananthapuram, Trivandrum, KL`
- `Tiruporur Clinical Private Limited | S.No 149/40, Rajiv Gandhi Salai, No.34 Omr Egattur, Chennai, Tiruporur, Kanchipuram, Tamil Nadu`  vs  `Tiruporur Homes Private Ltd | Tamil Nadu, S.NO 149/40, RAJIV GANDHI SALAI, NO.34 OMR EGATTUR, CHENNAI, TIRUPORUR`

### India: true matches retrieved but not selected

- `Guha Advisors | 841/24, Dev Nagar, Sonepat, Haryana`  vs  `Miraorbi | 841/24, Dev Nagar, Sonepat, Haryana`
- `Technology Archer Developers Private Limited | House No. 285, Rajendra Nagar, 1St Lane, Brundavangardens, Guntur, Andhra Pradesh`  vs  `Technology Archer Private Limited Center | House No. 85, Rajendra Nagar, 1St Lane, Brundavangardens, Guntur, AP`
- `India Voyage Switchgears | No 1211, Sri Mailarling Nilaya, 1St Main, 1St Cross 1St Phase, 5Th Stage Beml Layout, Rajarajeshwari Nagar Bengaluru, Bangalore, Karnataka`  vs  `India Vg Switchgears | No 1211, Bangalore, KA, Sri Mailarling Nilaya, 1St Main, 1St Cross 1St Phase, 5Th Stage Beml Layout, Rajarajeshwari Nagar Bengaluru`
- `Mill & Co | Camelot4Th Cross Ecc Road Whitefield, Bangalore, Karnataka`  vs  `Mill Co Services | `
- `KZI Smart Private Limited | 6Th Floor, Apple Towers, Nh-Bye Pass Palarivattom, Kochi, Ernakulam, Kerala`  vs  `KZI Smart Pirartgte Limited | KL, Ernakulam, H.no 5-6Th Floor, Apple Towers, Nh-bye Pass Palarivattom, Kochi`
- `Atri Steels (India) Pvt. Ltd. | #89/1, Bellanduru, Bangalore, Karnataka`  vs  `Smt Atri Steels-(India) Pvt Ltd | `
- `Praniya Construction Pvt Ltd | Flat No. 8C, Prop No. E-4, Defence Colony, New Delhi, South Delhi, Delhi`  vs  `Praniya Construction Pvt | Flat No. 8C/6, Nagar, Prop No. E-4, Defence Colony, DL`
- `New Delhi Health Private Limited | 81, Shankar Market Cannaught Circus, New Delhi, Central Delhi, Delhi`  vs  `Orbiriza | 81, New Delhi, Central Delhi, DL`
- `My Industries Private Limited | No 1624, 2Nd Floor, 30Th Cross, Hbr Lyt 5Th Block, 1St Stage, Brundava, N Nagar, Bangalore, Karnataka`  vs  `ಮೈ ಇಂಡಸ್ಟ್ರೀಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್ | NO 1624, BENGALURU, Karnataka`
- `Blue Food Private Limited | New Delhi, 401 Dwarka Towers R.No-611Nehru Place, Delhi, South Delhi`  vs  `BFPRIVATE.COM | 7-401 DWARKA TOWERS R.NO-611NEHRU PLACE, SOUTH DELHI, NEW DELHI CHITTARANJAN PARK, Delhi`
- `Sukh Travel Pvt Ltd | 3604 Street No 5 Near Arora Palace Opposite Dana Mandi Chet Singh Nagar Gil, L Road, Ludhiana, Punjab`  vs  `Sukh Pvt Ltd Services | `
- `Chip Public School Private Limited | 1159, 26Th Main, 4Th T Block, Jayanagar, Bangalore, Karnataka`  vs  `Chip Public School Private Limited | `

### India: true matches never retrieved

- `Bangalore Auto Pvt Ltd | B1-803, Mahendra Aarna, Kammasandra Road Ananthnagar, Electronic City Phase2, Bangalore, Karnataka`  vs  `Bangalore Pvt Ltd Center | KA, B1-803, Bangalore`
- `Media Tirupati Solar Private Limited | A-1/3 Ground Floor Paschim Vihar, Delhi, West Delhi, Delhi`  vs  `Media Media Tiropati Solar Private Limited | `
- `Mohali Mill LLP | D-27, Industrial Area Pb In Mohali, Punjab, Mohali, Rupnagar, Punjab`  vs  `M0hali Mill Llp | D-27, RUPNAGAR, NULL, Punjab`
- `Gold Engineering Limited | No. 18-A, 2Nd Phase, Peenya Industrial Area, Bangalore, Karnataka`  vs  `M/s Gold Engineering Ltd | 18-A, 2Nd Phase, Peenya Industrial Area, Bangalore, KA`
- `KG Consultant Private Limited | 3, Birbal Road, Ground Floor, Jangpura Extension, New Delhi, South Delhi, Delhi`  vs  `kgconsultant.com | 3, South Delhi, DL, New Delhi`
- `Surya Management Private Limited | 18-B1-B, 2Nd Phase, Doddenakundi Industrial Area Mahadevpura, Bangalore, Karnataka`  vs  `SURYA MANAGEMENT PRIVATE (LIMITED) | `
- `Aditya Construction Private Limited | M-517 D, Sector-M Lda Colony, Ashiyana, Lucknow, Uttar Pradesh`  vs  `aditya construction private 1imited | M-0517 D, Fatehpur, उत्तर प्रदेश`
- `Surya Foundation Limited | B2-818 Dlf Mypad, Vibhuti Khand, Lucknow, Uttar Pradesh`  vs  `Surya Limited Foundation | `
- `Santoshi Engineers Private Limited | 31-F, Ground Floor, Madan Park, Ashoka Park, New Delhi, West Delhi, Delhi`  vs  `Santoshi Engineers Privrate Limited | 21-F, West Delhi, New Delhi, DL`
- `Pragati Constructions (India) Partners | No. 27, 6Th Cross, 2Nd Stage, Manjunathnagar, Bangalore North, Bangalore, Karnataka`  vs  `M/s Pragati (India) Partners Center | `
- `Sree Red Properties Private Limited | Mumbai City, 18, Maharashtra, Kamal Niwas 394, N. C. Kelkar Road, Dadar West, Mumbai`  vs  `Sree Red Properties Limited Sérvices | 18, Mumbai, Mumbai City, महाराष्ट्र`
- `Tech Foundation | Hyderabad, Near Pochamma Temple Balanagar, 10-11-40 Main Road, Telangana, Ameerpet`  vs  `Tech Foundation | 0-11-40 Main Road, Hyderabad, TG`

### US: false links (selected, not a match)

- `First Ready Inc | 3905 Poplar Point Drive, Rockwall, TX`  vs  `FOURTH READY INC | 3905 POPLAR POINT DRIVE, ROCKWALL, TX`
- `Foot & Ankle American Group | 52 Jennifer Way, Gorham, ME`  vs  `Foot & Ankle American | `
- `Crestline Electronics LLC | 807 Schult Drive, Fayetteville, NC`  vs  `Crestline Electronics | `
- `Modern Plus Inc | 756 Stowe Street, Highlands Ranch, CO`  vs  `Modern Plús Inc | `
- `Rodriguez's Cafe | 170 William Street, Unit 820, Fall River, MA`  vs  `Rodriguez's Cafe Ltd | `
- `Fortin Generating LLC | 1295 W Circle Court, Troy, OH`  vs  `troyregionalsilver.com | Ohio, 295 W Circle Court, TROY CITY`
- `Brock's Colonial Vision Center | 809 New Jersey Avenue, Unit 1336, Washington, DC`  vs  `Vision Center Inc. | NEW JERSEY AVENUE, WASHINGTON, DC`
- `Advanced Desert Pgim | 614 Green Park, Nashville, TN`  vs  `Advanced Désert Pgim Co Co | `
- `Jaquith Strobel Vulcan Inc | 106 Cypress Point Place, Whitpain Township, PA`  vs  `The Jaquith Strobel Vulcan Inc | `
- `Physical Therapy Center | 1719 Vermont Avenue, Washington, DC`  vs  `Physical Therapy Center, Co | 1719C EAST CAPITOL STREET, WASHINGTON, DC`
- `Aguinaga Micro Inc | 3722 Echo Forest Trail, High Point, NC`  vs  `Aguinaga Micro Inc | `
- `Metropolitan Muniyield | 21631 Wave Hollow Drive, Cypress, TX`  vs  `Metropolitan Muniyield | `

### US: true matches retrieved but not selected

- `E & B Medical Inc | Tucson, AZ, 15025 Ajo Highway`  vs  `E & B Medical-Inc | `
- `Midwest Fresh Mozayyx Inc. | 11248 Emerald Avenue, Chicago, IL`  vs  `Midwest Fresh Mozayyx Inc | `
- `Jensen Chenghe Inc | 26 Hayden Rowe Street, Hopkinton, MA`  vs  `JAXFLUX | HOPKINTON, 26 Hayden Rowe Street, MA`
- `Zephola End LLC | 8534 Koufax Way, Florence, AZ`  vs  `Zephola End LLC | `
- `Cornerstone Flex LLC | 29889 Conquest Court, Unit Apt 10205, Huntsville, AL`  vs  `CORNERSTONE FLEX FLEX LLC | `
- `Swayne Short LLC | 8771 Navidad Court, Irving, TX`  vs  `SWAYNE SHORT-LLC | `
- `Munoz Trust Co | 2113 6th Street, Hempstead, NY`  vs  `Munoz Trust Co | `
- `32/12 Digitalbridge | VA, Prince William County, 7383 Roxbury Avenue`  vs  `32/12 Digitalbridge | `
- `Marita E. Girardot, DO, DDS PC | 1509 Gold Nugget Circle, Sioux Falls, SD`  vs  `Marita E. Girardot, DDS PC Center | `
- `Newton Highland Sterling LP | WI, 3649 Spring Court, Town Of Webb Lake`  vs  `newton highland sterling lp | 6649 Spring Court, Danbury, Wisconsin`
- `Duran, Nissie, P.A. | AL, Vernon, 164 1st Avenue`  vs  `Duran, Nissie, P.A. Co | `
- `Arnold, Salgado & Navar Wireless Inc | 404 Narrows Pointe Drive, Grasonville, MD`  vs  `Amold, Salgado & Navar | 404 Narrows Pointe Dr, Grasonville, Maryland`

### US: true matches never retrieved

- `Ace Live LLC | 103 Kenton Court, New Castle, DE`  vs  `Ace LLC Partners | www.acellcpa.com | `
- `Rocky Core Inc. | 24 Huffer Road, Parma, NY`  vs  `The Rocky Core Inc. | `
- `Perez School of Medicine LLC | 1318 Berrywood Lane, Houston, TX`  vs  `Perez School Of LLC Services | `
- `Dermatology Silver Physicians | 11 Dogwood Drive, Poughkeepsie, NY`  vs  `Dermatology Silver #24419 | `
- `Powers Diana Center | 317 Milwaukee Street, Houston, TX`  vs  `Powers Diana Cntrr | `
- `Schenck, Harmon and Curtis LLC | 12424 3rd Avenue, Spokane Valley, WA`  vs  `Schenck, Harmon and | `
- `Dental Total Care | 55 Biddie Lane, Lost Creek, WV`  vs  `Dental T0tal Cáre | `
- `Con's Advisors PLLC | 6204 Rd 15, Centerburg, OH`  vs  `Con's Adviesrls PLLC | 5204 RD 15, S BLOOMFIELD TWP, OH`
- `Kestex Produce, Inc | 132-17 Metropolitan Avenue, Richmond Hill, NY`  vs  `Kestex (Produce,) | www.kestexpr.com | `
- `Emmalynne's Silver Investments | 7125 Fair Oaks Boulevard, Unit APT 210, Carmichael, CA`  vs  `EMMALYNNE'S SILVER IVFESSMENTS | `
- `Aspen Bitwise | 2339 Three B's & K Road, Sunbury, OH`  vs  `Aspen-8itwise | `
- `Valley Center | 663 Old Plantation Drive, Campbell County, VA`  vs  `Valley Center Corp | `
