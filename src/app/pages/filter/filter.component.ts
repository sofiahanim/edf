import { Component, OnInit, Input, EventEmitter,Output } from '@angular/core';

@Component({
  selector: 'ngx-filter',
  templateUrl: './filter.component.html',
  styleUrls: ['./filter.component.scss']
})
export class FilterComponent implements OnInit {

  constructor() { }

  ngOnInit(): void {
  }
  @Input('total') all: number = 0;
  @Input() free: number = 0;
  @Input() premium: number = 0;

  selectedValue: string= 'All';

  @Output()
  filterSelectionChanged: EventEmitter<string> = new EventEmitter<string>();

  onSelectionChange(){
    this.filterSelectionChanged.emit(this.selectedValue);
    //console.log(this.selectedValue);
  }

  
}
